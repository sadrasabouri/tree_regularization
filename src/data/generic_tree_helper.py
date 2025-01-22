from datasets import Dataset as HFDataset
import random
from tqdm import tqdm
from nltk import Tree
from datasets import load_dataset
import json
import torch
import pdb
from data.data_utils import reformat_tree, flatten, binarize_tree, get_word_boundaries, get_word_boundaries_sow, tree_to_parse_decisions, reformat_tree_generic
from transformers import GPT2Tokenizer

def build_generic_tree_dataset(
    data_file_given=None,
    data_ratio = 1.0,
    in_vocab = None,
    hf = False
):
    # 1. word boundaries come from tree now
    # 2. trees need to be binarized no matter what
    # 3. reformatting tree removes word boundary info

    def read_data(splits):
        in_sentences = []
        parses = []
        word_boundaries = []
        index_map = {split: [] for split in splits}
        for split in splits:
            split_file = split

            if data_file_given is None:
                data_file = "bllip-lg-depth"
            else:
                data_file = data_file_given

            with open(
                "{}/{}.txt".format(
                    data_file,
                    split_file,
                ),
                "r",
            ) as reader:
                print("Reading trees for {}".format(split_file))
                data = [
                    Tree.fromstring(l.strip()) for l in tqdm(reader.readlines()[:(int(data_ratio * 1755715) if data_ratio < 1.0 else -1)])
                ]

            exceptions = 0
            for sent in tqdm(data):
                # print(sent)
                try:
                    curr_word_boundary = get_word_boundaries(flatten(sent, add_eos=False, clean=False), in_vocab, hf=hf)
                    reformatted_sent = reformat_tree_generic(sent, in_vocab, hf=hf) # this will retokenize given NLTK trees
                    curr_sent = flatten(reformatted_sent, add_eos=False, clean=False, separator = "")
                    reformatted_sent.chomsky_normal_form()
                    index_map[split].append(len(in_sentences))
                    in_sentences.append(curr_sent)
                    parses.append(reformatted_sent)
                    word_boundaries.append(curr_word_boundary)
                except:
                    continue

        return in_sentences, parses, word_boundaries, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, parses, word_boundaries, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    dataset = {}
    for split in splits:
        # DEBUG
        in_subset = get_subset(in_sentences, index_map[split])
        in_parses = get_subset(parses, index_map[split])
        in_word_boundaries = get_subset(word_boundaries, index_map[split])

        if hf:
            in_subset_tokenized = [in_vocab(s)["input_ids"] for s in in_subset] # remove sos
        else:
            in_subset_tokenized = [in_vocab(s) for s in in_subset]

        in_lens = [len(s) for s in in_subset_tokenized]

        # compute word boundaries and tokenized words for treereg
        in_subset_boundaries = [" ".join([str(_) for _ in word_boundaries]) for word_boundaries in in_word_boundaries]
        # pdb.set_trace()

        parse_dicts = []
        for p in tqdm(in_parses):
            parse_dict = {}
            _ = tree_to_parse_decisions(p, 0, parse_dict, check=False)
            parse_dicts.append(json.dumps(parse_dict))

        data = {
            "in": in_subset_tokenized,
            "in_len": in_lens,
            "idxs": index_map[split],
            "parses": parse_dicts,
            "word_boundaries": in_subset_boundaries
        }
        # pdb.set_trace()

        dataset_curr = HFDataset.from_dict(data)
        dataset[split] = dataset_curr

    return dataset, in_vocab, in_sentences
