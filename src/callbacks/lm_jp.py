from transformers import GPT2Tokenizer
import torch
import numpy as np
import pickle 
from nltk import Tree
from data.data_utils import reformat_tree, flatten, get_word_boundaries, get_word_boundaries_sow, reformat_tree_generic, binarize_tree
from callbacks.eval_utils import eval_base_model, convert_tree_to_tuple_and_collate, get_parsing_accuracy, make_preds_base_model, convert_tree_to_tuple
from tqdm import tqdm
import json
import pdb

def callback_jp(model, in_vocab, split, regularizer = None, data_folder_given=None, hf=False, data_ratio = 1.0, layer_id = -1, sci_heads = -1.):
    """Callback function on BLIMP for pushdown lm training."""
    if data_folder_given:
        folder_dir = data_folder_given
    else:
        folder_dir = "bllip-lg"
    
    with open("{}/{}.txt".format(folder_dir, split)) as f:
        if (split == 'train'):
            data = [Tree.fromstring(l.strip()) for l in f.readlines()[:min(int(data_ratio * 1755715), 3000)]]
        else:
            data = [Tree.fromstring(l.strip()) for l in f.readlines()]

    examples = [(reformat_tree_generic(d, in_vocab, hf=hf), flatten(reformat_tree_generic(d, in_vocab, hf=hf), add_eos=False, clean=False, separator = "")) for d in data]
    word_boundaries = [get_word_boundaries(flatten(d, add_eos=False, clean=False), in_vocab, hf=hf) for d in data]

    sent_ppl, hidden_states = eval_base_model(model, examples, in_vocab, device=torch.device('cuda:0'), hf=hf, get_hidden_states = regularizer is not None, layer_id = layer_id, sci_heads = sci_heads)

    # get parsevals
    actual = 0
    if regularizer is not None:
        gold_parses = []
        predicted_parses = []
        for idx, (d, sentence) in tqdm(enumerate(examples)):
            actual += 1
            model.eval()
            curr_word_boundaries = word_boundaries[idx]
            chart = regularizer.build_chart(hidden_states[idx].unsqueeze(0), [curr_word_boundaries], None)
            predicted_parse, _ = regularizer.get_parse([sentence], chart, [curr_word_boundaries], separator="")
            predicted_parses.append(predicted_parse[0])
            gold_parses.append(convert_tree_to_tuple(data[idx]))

        parsing_acc = get_parsing_accuracy(predicted_parses, gold_parses, split)
    
    # Get accs for BLIMP
    acc = 0
    if split == "val":
        blimp_examples = []

        with open("src/callbacks/validated_minimal_pairs.jsonl", "r") as f:
            for line in f:
                example = json.loads(line)
                blimp_examples.append(example["good_sentence"])
                blimp_examples.append(example["bad_sentence"])

        all_sent_logprobs, _ = make_preds_base_model(
                model, in_vocab, blimp_examples, hf=hf, device=torch.device('cuda:0')
            )

        num_pairs = len(blimp_examples)//2
        for num in range(num_pairs):
            good_prop = np.sum(all_sent_logprobs[2*num])/len(all_sent_logprobs[2*num])
            bad_prop = np.sum(all_sent_logprobs[2*num + 1])/len(all_sent_logprobs[2*num + 1])

            if (good_prop >= bad_prop):
                acc += 1

        acc /= num_pairs

    if regularizer is not None:
        return {"ppl": sent_ppl,
                "blimp_acc": acc,
                "parsing_acc": parsing_acc["f1"]}
    else:
        return {"ppl": sent_ppl, "blimp_acc": acc}