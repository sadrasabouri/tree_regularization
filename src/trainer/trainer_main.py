from sympy import use
import torch
from tqdm import tqdm
import os
import wandb
import time
import math

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs

from transformers.data.data_collator import DataCollatorWithPadding
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

import trainer.collate as collate
import wandb
import json
import pdb
import util
from itertools import chain

def get_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

def get_opt(lr, weight_decay, model):
    if type(model) != torch.nn.Module:
        model = model.model
    no_decay = ["bias", "LayerNorm.weight"]
    adam_epsilon = 1e-7
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=adam_epsilon,
    )
    return optimizer

def get_scheduler(opt, start_lr, min_lr, t_total):
    # cosine scheduler from 6e-4 to 6e-5
    num_warmup_steps = (5 * t_total) // 1000 # 0.5% of total

    def get_lr(it):
        if it < num_warmup_steps:
            return it / num_warmup_steps
        
        decay_ratio = (it - num_warmup_steps) / (t_total - num_warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr/start_lr + coeff * (1 - min_lr/start_lr)

    return LambdaLR(opt, get_lr, -1)

def train_loop(
    args,
    model,
    train_dataset,
    parse_dataset,
    device,
    callback_fn,
    regularizer = None
):
    train_batch_size = args.batch_size
    accum_steps = args.accum_steps
    eval_every = args.eval_every
    max_steps = args.max_train_steps
    regularizer_steps = args.regularizer_steps
    use_packing = args.pack
    max_seq_len = args.max_seq_len
    save_interval = args.save_interval
    save_dir = args.save_dir
    treereg_same_data = args.treereg_same_data

    opt = get_opt(args.start_lr, args.weight_decay, model)
    scheduler = get_scheduler(opt, args.start_lr, args.end_lr, max_steps)
    train_data_collator = collate.VarLengthCollate(None)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    if args.pack:
        # this dataset handles all packing
        train_dataset = util.PackedDataset(train_dataset, max_seq_len, model.encoder_sos)

    model.model, opt, scheduler = accelerator.prepare(model.model, opt, scheduler)

    best_val_ppl = 1000000
    num_steps = 0

    while True:
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        )

        if use_packing:
            # this dataset handles all packing
            parse_dataset = util.PackedDataset(parse_dataset, max_seq_len, model.encoder_sos)

        parse_dataloader = iter(DataLoader(
            parse_dataset,
            sampler=RandomSampler(parse_dataset, replacement=True, num_samples=100000000),
            batch_size=train_batch_size,
            collate_fn=train_data_collator,
        ))

        train_dataloader = accelerator.prepare(train_dataloader)
        parse_dataloader = accelerator.prepare(parse_dataloader)

        total_train_size = len(train_dataset)
        if num_steps > max_steps:
            break

        with torch.enable_grad(), tqdm(total=total_train_size, disable=not accelerator.is_local_main_process) as progress_bar:
            sci_scores_agg = []
            losses = []

            for curr_batch_dict in train_dataloader:
                model.model.train()

                # curr_batch_dict_gpu = {}
                # for key in curr_batch_dict:
                #     if (key in ['string', 'parses', 'idxs', 'contained_exs', 'word_boundaries']):
                #         curr_batch_dict_gpu[key] = curr_batch_dict[key]
                #     else:
                #         curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)

                # with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=True):
                    # out = model(curr_batch_dict_gpu, pack=use_packing, nli=True if args.dataset == "multinli" else False, 
                                                            # regularize=args.regularize and treereg_same_data, layer_id=args.layer_id, sci_heads=args.sci_heads)
                out = model(curr_batch_dict, pack=use_packing, nli=True if args.dataset == "multinli" else False, 
                                                            regularize=args.regularize and treereg_same_data, layer_id=args.layer_id, sci_heads=args.sci_heads)
                    
                loss_curr = out.loss
                hidden_states = out.hidden_states
                loss_curr /= accum_steps
                if accelerator.is_local_main_process:
                    losses.append(loss_curr.item())
                
                accelerator.backward(loss_curr, retain_graph=treereg_same_data)

                if accelerator.is_local_main_process:
                    progress_bar.update(curr_batch_dict["in"].shape[1])

                if num_steps % regularizer_steps == 0 and args.regularize:
                    # Sample a batch from the parse dataloader
                    if not treereg_same_data:
                        parse_batch_dict = next(parse_dataloader)
                        # curr_parse_dict_gpu = {}
                        # for key in curr_batch_dict:
                        #     if (key in ['string', 'parses', 'idxs', 'contained_exs', 'word_boundaries']):
                        #         curr_parse_dict_gpu[key] = parse_batch_dict[key]
                        #     else:
                        #         curr_parse_dict_gpu[key] = parse_batch_dict[key].to(device)
                        # out = model(curr_parse_dict_gpu, pack=use_packing, nli=True if args.dataset == "multinli" else False, 
                        #                                     regularize=args.regularize, layer_id=args.layer_id, sci_heads=args.sci_heads)
                        out = model(parse_batch_dict, pack=use_packing, nli=True if args.dataset == "multinli" else False, 
                                                             regularize=args.regularize, layer_id=args.layer_id, sci_heads=args.sci_heads)
                        hidden_states = out.hidden_states
                    else:
                        parse_batch_dict = curr_batch_dict

                    curr_parses = parse_batch_dict['parses']
                    curr_word_boundaries = parse_batch_dict['word_boundaries']
                    if use_packing:
                        curr_parses = list(chain(*curr_parses))
                        curr_word_boundaries = list(chain(*curr_word_boundaries))
                        # extract the correct hidden states
                        hidden_states = util.extract_hidden_states(hidden_states, curr_word_boundaries)
                    curr_word_boundaries = [[int(_) for _ in wb.split(" ")] for wb in curr_word_boundaries]
                    curr_parses = [json.loads(_) for _ in curr_parses]
                    hidden_states = hidden_states[:,1:,:] # skip the hidden states for SOS                
                    scin_charts = regularizer.build_chart(hidden_states, curr_word_boundaries, curr_parses)

                    # with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16, enabled=True):
                    treereg_scores, broken_acc = regularizer.get_score(scin_charts, curr_word_boundaries, curr_parses, accelerator.device)
                    print(broken_acc)

                    fin_treereg_scores = torch.mean(torch.stack(treereg_scores))
                    wandb.log({"fin_treereg_scores": fin_treereg_scores.item(), "train_parseval": broken_acc})
                    sci_loss = -fin_treereg_scores/accum_steps
                    sci_scores_agg.append(fin_treereg_scores.item()/accum_steps)
                    accelerator.backward(sci_loss)
                    # sci_loss.backward()

                if len(losses) == accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), 1.0
                    )

                    if accelerator.is_local_main_process:
                        progress_bar.set_postfix(
                            {"loss": sum(losses), "num_steps": num_steps}
                        )
                        grad_norm = get_grad_norm(model.model)

                        if (args.regularize and num_steps % regularizer_steps == 0):
                            if broken_acc is not None:
                                wandb.log(
                                    {
                                        "bracket_acc_train": broken_acc
                                    }
                                )
                            wandb.log(
                                {
                                    "sci_score": sum(sci_scores_agg)
                                }
                            )
                        
                        wandb.log(
                            {
                                "loss": sum(losses),
                                "grad_norm": grad_norm,
                                "iteration": num_steps
                            }
                        )

                    opt.step()
                    scheduler.step()
                    # model.model.zero_grad(set_to_none=True)
                    opt.zero_grad()
                    losses = []
                    sci_scores_agg = []
                    num_steps += 1

                    if accelerator.is_local_main_process and num_steps%save_interval == 0:
                        save_path = f"{os.path.join(save_dir, 'state')}_{num_steps}.pt"
                        unwrapped_model = accelerator.unwrap_model(model.model)
                        accelerator.save(unwrapped_model.state_dict(), save_path)
                        # torch.save(model.model.state_dict(), save_path)
                        print(f"Saved model at step {num_steps} to {save_path}")

                    if num_steps%eval_every == 0 and accelerator.is_local_main_process:
                        print("Evaluating at step {}".format(num_steps))

                        train_score = callback_fn("train", regularizer, args)
                        val_score = callback_fn("val", regularizer, args)
                        test_score = callback_fn("test", regularizer, args)
                        print(val_score)
                        if (val_score['ppl'] < best_val_ppl):
                            best_val_ppl = val_score['ppl']
                            best_blimp = val_score['blimp_acc']
                            save_path = f"{os.path.join(save_dir, 'state')}_best_model.pt"
                            torch.save(model.model.state_dict(), save_path)
                            print(f"Saved best model to {save_path}")

                        if (args.regularize):
                            wandbdict = {
                                "train_parseval": train_score['parsing_acc'],
                                "val_ppl": val_score['ppl'],
                                "test_ppl": test_score['ppl'],
                                "blimp_score": val_score['blimp_acc'],
                                "best_blimp_score": best_blimp,
                                "val_parseval": val_score['parsing_acc'],
                                "test_parseval": test_score['parsing_acc'],
                            }
                        else:
                            wandbdict = {
                                "iteration": num_steps,
                                "val_ppl": val_score['ppl'],
                                "test_ppl": test_score['ppl'],
                                "blimp_score": val_score['blimp_acc'],
                                "best_blimp_score": best_blimp
                            }

                        if args.dataset in ['nmonli', 'snli', "multinli"]:
                            wandbdict['monli_acc']= val_score["monli_acc"]
                        wandb.log(wandbdict)
                    
                if num_steps > max_steps:
                    break
            
            if losses and accelerator.is_local_main_process:
                progress_bar.set_postfix({"loss": sum(losses), "num_steps": num_steps})
                grad_norm = get_grad_norm(model.model)
                wandb.log(
                    {
                        "loss": sum(losses),
                        "grad_norm": grad_norm,
                        "iteration": num_steps,
                    }
                )

                opt.zero_grad()
                losses = []
                if num_steps > max_steps:
                    break   
    
    # save_path = f"{os.path.join(save_dir, 'state')}_final_model.pt"
    # torch.save(model.model.state_dict(), save_path)
    # print(f"Saved final model to {save_path}")

    if accelerator.is_local_main_process:
        save_path = f"{os.path.join(save_dir, 'state')}_final_model.pt"
        unwrapped_model = accelerator.unwrap_model(model.model)
        accelerator.save(unwrapped_model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")

    return