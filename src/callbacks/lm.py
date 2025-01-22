from transformers import GPT2Tokenizer
import torch
import numpy as np
import pickle 
from nltk import Tree
from data.data_utils import reformat_tree, flatten, get_word_boundaries, get_word_boundaries_sow
from callbacks.eval_utils import eval_base_model, convert_tree_to_tuple_and_collate, get_parsing_accuracy, make_preds_base_model
from tqdm import tqdm
import json
import pdb

def callback_lm(model, in_vocab, split, regularizer = None, data_folder_given=None, hf=False, data_ratio = 1.0, layer_id = -1, sci_heads = -1.):
    """Callback function on BLIMP for pushdown lm training."""
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg-depth"
    
    with open("{}/{}.txt".format(folder_dir, split)) as f:
        if (split == 'train'):
            data = [Tree.fromstring(l.strip()) for l in f.readlines()[:min(int(data_ratio * 1755715), 3000)]]
        else:
            data = [Tree.fromstring(l.strip()) for l in f.readlines()]

    if hf:
        examples = [(reformat_tree(d, in_vocab, True), flatten(d, add_eos=False, clean=True)) for d in data]
    else:
        examples = [(d, flatten(d, add_eos=False)) for d in data]

    sent_ppl, hidden_states = eval_base_model(model, examples, in_vocab, device=torch.device('cuda:0'), hf=hf, get_hidden_states = regularizer is not None, layer_id = layer_id, sci_heads = sci_heads)

    # get parsevals
    actual = 0
    if regularizer is not None:
        gold_parses = []
        predicted_parses = []
        for idx, (d, sentence) in enumerate(examples):
            actual += 1
            model.eval()
            if hf:
                curr_word_boundaries = get_word_boundaries(sentence, in_vocab)
            else:
                curr_word_boundaries = get_word_boundaries_sow(sentence)
            chart = regularizer.build_chart(hidden_states[idx].unsqueeze(0), [curr_word_boundaries], None)
            predicted_parse, _ = regularizer.get_parse([flatten(d, add_eos=False)], chart, [curr_word_boundaries])
            predicted_parses.append(predicted_parse[0])
            gold_parses.append(convert_tree_to_tuple_and_collate(d, curr_word_boundaries))

        parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses, split)
    
    # Get accs for BLIMP
    acc = 0
    if split == "val":
        if hf:
            with open("/afs/cs.stanford.edu/u/ananjan/tree_regularization/src/callbacks/blimp.pkl", "rb") as f:
                blimp_data = [reformat_tree(Tree.fromstring(t), in_vocab, True) for t in pickle.load(f)]
            blimp_examples = [flatten(d, add_eos=False, clean=True) for d in blimp_data]
        else:
            with open("/afs/cs.stanford.edu/u/ananjan/tree_regularization/src/callbacks/blimp.pkl", "rb") as f:
                blimp_data = [Tree.fromstring(t) for t in pickle.load(f)]
            blimp_examples = [flatten(d, add_eos=False) for d in blimp_data]

        all_sent_logprobs, _ = make_preds_base_model(
                model, in_vocab, blimp_examples, hf=hf, device=torch.device('cuda:0')
            )

        num_pairs = len(blimp_examples)//2
        for num in range(num_pairs):
            good_prop = np.sum(all_sent_logprobs[num])/len(all_sent_logprobs[num])
            bad_prop = np.sum(all_sent_logprobs[num + num_pairs])/len(all_sent_logprobs[num + num_pairs])

            if (good_prop >= bad_prop):
                acc += 1

        acc /= num_pairs

    if regularizer is not None:
        return {"ppl": sent_ppl,
                "blimp_acc": acc,
                "parsing_acc": parsing_acc["f1"]}
    else:
        return {"ppl": sent_ppl, "blimp_acc": acc}