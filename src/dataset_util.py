from data.bllip_helper import build_bllip_dataset
from data.generic_tree_helper import build_generic_tree_dataset
from callbacks.lm import callback_lm
from callbacks.lm_jp import callback_jp
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
import pickle

BASE_DIR = "/project2/jonmay_1455/sadra/project/tree_regularization_sadra"

def dataset_helper(dataset_name, args):
    if dataset_name == "jp-alt":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open(f'{BASE_DIR}/src/data/jp.pkl', 'rb'))
        datasets, _, _ = build_generic_tree_dataset(
                data_file_given=f'{BASE_DIR}/datasets/jp-ninjal',
                hf = args.hf,
                in_vocab = in_vocab
            )
    elif dataset_name == "bllip-lg":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open(f'{BASE_DIR}/src/data/blimp_vocab.pkl', 'rb'))
        datasets, _, _ = build_bllip_dataset(
                data_file_given=f'{BASE_DIR}/datasets/bllip-lg',
                hf = args.hf,
                in_vocab = in_vocab,
                gpt2 = args.nanogpt
            )
    elif dataset_name == "bllip-md":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open(f'{BASE_DIR}/src/data/blimp_vocab.pkl', 'rb'))
        datasets, _, _ = build_bllip_dataset(
                data_file_given=f'{BASE_DIR}/datasets/bllip-lg',
                hf = args.hf,
                data_ratio = 0.001,
                in_vocab = in_vocab,
                gpt2 = args.nanogpt
            )
    elif dataset_name == "bllip-sm":
        if args.hf:
            in_vocab = AutoTokenizer.from_pretrained(args.hf_model_name)
        else:
            in_vocab = pickle.load(open(f'{BASE_DIR}/src/data/blimp_vocab.pkl', 'rb'))
        datasets, _, _ = build_bllip_dataset(
                data_file_given=f'{BASE_DIR}/datasets/bllip-lg',
                hf = args.hf,
                data_ratio = 0.1,
                in_vocab = in_vocab,
                gpt2 = args.nanogpt
            )
    else:
        raise NotImplementedError()

    return datasets, in_vocab

def get_callback_fn(args, model, in_vocab):
    """
    Returns the appropriate callback function based on the dataset type specified in args.

    Args:
        args: Command line arguments.
        language_model (bool): Flag to determine if it's a language model.
        model: The trained model.
        in_vocab (CharVocabulary): Input vocabulary.
        datasets: The datasets used for training and evaluation.

    Returns:
        function: The corresponding callback function.
    """

    # CHANGES REQUIRED HERE
    if not args.callback:
        return None

    dataset_callbacks = {
        "jp-alt": lambda split, regularizer, args: callback_jp(model, in_vocab, split, regularizer, 
            data_folder_given=f'{BASE_DIR}/datasets/jp-ninjal', hf=args.hf, data_ratio = 1.0, layer_id=args.layer_id, sci_heads=args.sci_heads),
        "bllip-lg": lambda split, regularizer, args: callback_lm(model, in_vocab, split, regularizer, 
            data_folder_given=f'{BASE_DIR}/datasets/bllip-lg', hf=args.hf, data_ratio = 1.0, layer_id=args.layer_id, sci_heads=args.sci_heads),
        "bllip-md": lambda split, regularizer, args: callback_lm(model, in_vocab, split, regularizer, 
            data_folder_given=f'{BASE_DIR}/datasets/bllip-lg', hf=args.hf, data_ratio = 0.001, layer_id=args.layer_id, sci_heads=args.sci_heads),
        "bllip-sm": lambda split, regularizer, args: callback_lm(model, in_vocab, split, regularizer,
            data_folder_given=f'{BASE_DIR}/datasets/bllip-lg', hf=args.hf, data_ratio = 0.1, layer_id=args.layer_id, sci_heads=args.sci_heads),
    }

    return dataset_callbacks.get(args.dataset, lambda split: Exception("Invalid dataset"))