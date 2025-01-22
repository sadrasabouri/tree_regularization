from datasets import Dataset as HFDataset
import random
from tqdm import tqdm
from nltk import Tree
from datasets import load_dataset
import json
import torch
import pdb
from data.data_utils import reformat_tree, flatten, binarize_tree, get_word_boundaries, get_word_boundaries_sow, tree_to_parse_decisions
from transformers import GPT2Tokenizer

def build_bllip_dataset(
    data_file_given=None,
    data_ratio = 1.0,
    in_vocab = None,
    hf = False,
    gpt2 = False
):
    def read_data(splits):
        in_sentences = []
        parses = []
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

            for sent in tqdm(data):
                if hf:
                    sent = reformat_tree(sent, in_vocab, True, gpt2 = gpt2) # this will retokenize the GPT-2 tokenized trees

                index_map[split].append(len(in_sentences))
                in_sentences.append(flatten(sent, add_eos=False, clean=hf))

                if not isinstance(sent, Tree):
                    parses.append(binarize_tree(sent))
                else:
                    parses.append(sent)

        return in_sentences, parses, index_map

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    splits = ["train", "val", "test"]
    in_sentences, parses, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        in_parses = get_subset(parses, index_map[split])

        if hf:
            in_subset_tokenized = [in_vocab(s)["input_ids"][1:] if not gpt2 else list(in_vocab(s)["input_ids"]) for s in in_subset] # remove sos
        else:
            in_subset_tokenized = [in_vocab(s) for s in in_subset]

        in_lens = [len(s) for s in in_subset_tokenized]

        # compute word boundaries and tokenized words for treereg
        if hf:
            in_subset_boundaries = [" ".join([str(k) for k in get_word_boundaries(_, in_vocab)]) for _ in in_subset]
        else:
            in_subset_boundaries = [" ".join([str(k) for k in get_word_boundaries_sow(_)]) for _ in in_subset]

        parse_dicts = []
        for p in tqdm(in_parses):
            parse_dict = {}
            _ = tree_to_parse_decisions(p, 0, parse_dict)
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
