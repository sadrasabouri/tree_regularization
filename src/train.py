from argument_parser import get_parser
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from trainer.trainer_main import train_loop
from regularizer.regularizer_main import TreeRegularizer
from model_util import create_model, create_lm, create_model_interface
from dataset_util import dataset_helper, get_callback_fn
from util import working_dir
from vocabulary import CharVocabulary

torch.set_printoptions(threshold=10000)

def get_base_transformer_lm(args, in_vocab: CharVocabulary, model_load_path: str = None):
    """
    Returns a base transformer language model and its interface.

    Args:
        args: Command line arguments.
        in_vocab (CharVocabulary): Input vocabulary.
        model_load_path (str, optional): Path to the pre-trained model. Defaults to None.

    Returns:
        tuple: A tuple containing the model and its interface.
    """
    model = create_lm(len(in_vocab), args.vec_dim,
                      args.n_heads, args.encoder_n_layers, args.embedding_dropout, args.output_dropout, args.relative, activation=F.leaky_relu)
    if model_load_path:
        print(f"INFO: Loading pretrained model from {model_load_path}")
        model.load_state_dict(torch.load(
            model_load_path, map_location=torch.device("cpu")))
    interface = create_model_interface(model, is_lm=True)
    return model, interface

def get_base_transformer_hf(args, in_vocab: CharVocabulary, model_load_path: str = None):
    """
    Returns a base transformer language model and its interface.

    Args:
        args: Command line arguments.
        in_vocab (CharVocabulary): Input vocabulary.
        model_load_path (str, optional): Path to the pre-trained model. Defaults to None.

    Returns:
        tuple: A tuple containing the model and its interface.
    """
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_name)

    if model_load_path:
        model.load_state_dict(torch.load(
            model_load_path, map_location=torch.device("cpu")))
   
    interface = create_model_interface(model, is_lm=True, hf=True, in_vocab=in_vocab)
    return model, interface

def get_datasets_and_vocab(args):
    """
    Retrieves datasets and vocab based on the dataset type specified in args.

    Args:
        args: Command line arguments.

    Returns:
        tuple: A tuple containing the datasets and input vocabulary.
    """
    
    datasets, in_vocab = dataset_helper(args.dataset, args)

    if args.parse_dataset == 'None':
        parse_datasets = datasets
    else:
        parse_datasets, _ = dataset_helper(args.parse_dataset, args)
    
    return datasets, parse_datasets, in_vocab

def set_seed(args):
    """
    Sets random seeds for reproducibility.

    Args:
        args: Command line arguments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def init_wandb(args):
    """
    Initializes the wandb environment.

    Args:
        args: Command line arguments.
    """
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity)
    wandb.run.name = f"{args.save_dir}-{args.seed}"
    wandb.run.save()

def get_datasets_and_vocab(args):
    """
    Retrieves datasets and vocab based on the dataset type specified in args.

    Args:
        args: Command line arguments.
        language_model (bool): Flag to determine if it's a language model.

    Returns:
        tuple: A tuple containing the datasets and input vocabulary.
    """
    
    datasets, in_vocab = dataset_helper(args.dataset, args)

    if args.parse_dataset == 'None':
        parse_datasets = datasets
    else:
        parse_datasets, _ = dataset_helper(args.parse_dataset, args)
    

    return datasets, parse_datasets, in_vocab

def get_regularizer(args):
    """
    Get tree projection regularizer if required.

    Args:
        args: Command line arguments.
    """
    if (args.regularize):
        regularizer = TreeRegularizer(args.orth_bidir)
    else:
        regularizer = None

    return regularizer

def main_lm(args):
    """
    Main function for language modeling tasks.

    Args:
        args: Command line arguments.
    """

    datasets, parse_datasets, in_vocab = get_datasets_and_vocab(args)

    regularizer = get_regularizer(args)

    if args.hf:
        model, interface = get_base_transformer_hf(
            args, in_vocab, model_load_path=args.model_load_path)
    else:
        model, interface = get_base_transformer_lm(
            args, in_vocab, model_load_path=args.model_load_path)

    callback_fn = get_callback_fn(
        args, model, in_vocab)

    device = torch.device(f"cuda:{args.gpu_id}")
    model.to(device)
    if args.save_dir:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    eval_keys = ["val", "test"]

    train_loop(
        args,
        interface,
        datasets["train"],
        parse_datasets["train"],
        device,
        callback_fn=callback_fn,
        regularizer = regularizer
    )

if __name__ == "__main__":
    # test out HF integration
    # write loader for code (generic loader)
    parser = get_parser()
    args = parser.parse_args()

    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity)
    wandb.run.name = f"{args.save_dir}-{args.seed}"
    wandb.run.save('wandb_logs/*')
    main_lm(args)
