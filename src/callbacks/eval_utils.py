import torch
import trainer.collate as collate
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import pdb

def logsumexp(x):
    x = np.array(x)
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def eval_base_model(lm, examples, tokenizer, device, hf=False, get_hidden_states=False,layer_id=-1,sci_heads=-1.):
    """Evaluate a standard transformer LM (no pushdown / external stack)."""
    all_sent_logprobs, hidden_states = make_preds_base_model(
        lm, tokenizer, [s for p, s in examples], device, hf=hf, get_hidden_states=get_hidden_states,layer_id=layer_id,sci_heads=sci_heads
    )
    sent_ppl = compute_perplexity_from_logprobs([x for x in all_sent_logprobs])
    return sent_ppl, hidden_states

def compute_per_token_logprob(eos_token, str_logits, inputs, input_lens):
    str_logprobs = []
    # (bs x len x vocab)
    str_logits = str_logits.transpose(0, 1)
    eos_token = torch.tensor([eos_token]).to(inputs.device)
    for idx, (c_input, str_logprob) in enumerate(zip(inputs, str_logits)):
        curr_len = input_lens[idx]
        ## len x vocab
        ### shift input by 1 to evaluate LM
        target = torch.cat([c_input[1:curr_len], eos_token])
        eos_removed_logits = str_logprob[:curr_len]
        # eos_removed_logits = str_logprob
        eos_logprobs = F.log_softmax(eos_removed_logits, dim=1)
        logprobs_curr = torch.gather(eos_logprobs, 1, target.unsqueeze(1)).squeeze(1)
        str_logprobs.append(logprobs_curr.cpu().numpy())
    return str_logprobs

def tokenizer_helper(
    tokenizer,
    data_collator,
    inp_slice
):
    inp_list = [tokenizer(s) for s in inp_slice]
    in_lens = [len(s) for s in inp_list]
    inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
    inp = data_collator(inp_to_collate)
    in_len = inp["in_len"].long()
    return (
        inp["in"].transpose(0, 1),
        in_len,
    )

def pad_tensor(tensor, max_dim):
    padding = (0, 0, 0, max_dim - tensor.shape[1])
    return F.pad(tensor, padding, "constant", 0)

@torch.no_grad()
def make_preds_base_model(
    lm, tokenizer, sents, device, hf=False, get_hidden_states=False,layer_id=-1,sci_heads=-1.
):
    """
    Use language model to make predictions on the given sentences.
    """

    data_collator = collate.VarLengthCollate(None)
    batch_size = 32
    st = 0
    device = device
    all_sent_logprobs = []
    all_hidden_states = []

    def tokenizer_add(s):
        if hf:
            return tokenizer.encode(s)
        else:
            return [lm.encoder_sos] + tokenizer(s)
    
    def generate_mask(max_len, in_lens):
        return torch.arange(max_len).expand(len(in_lens), max_len).to(in_lens.device) < in_lens.unsqueeze(1)

    while st < len(sents):
        en = min(len(sents), st + batch_size)
        sent_slice = sents[st:en]
        inputs, input_lens = tokenizer_helper(
            tokenizer_add, data_collator, sent_slice
        )
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)

        if hf:
            attn_mask = generate_mask(inputs.shape[1], input_lens).to(inputs.device)
            outputs = lm(inputs, attention_mask=attn_mask, output_hidden_states=get_hidden_states)
            all_str_logits_curr = outputs.logits
            if get_hidden_states:
                proportion = int(sci_heads * outputs.hidden_states[0].shape[-1])
                curr_hidden_states = outputs.hidden_states[layer_id][:, :, :proportion]
                curr_hidden_states = curr_hidden_states[:,1:,:]
                all_hidden_states.append(curr_hidden_states)
        else:
            outputs = lm(inputs, input_lens, get_hidden_states = get_hidden_states, layer_id = layer_id, sci_heads = sci_heads)
            all_str_logits_curr = outputs.data
            if get_hidden_states:
                all_hidden_states.append(outputs.hidden_states[:,1:,:])

        logprobs_curr = compute_per_token_logprob(
            tokenizer(tokenizer.eos_token)["input_ids"][1] if hf else lm.encoder_eos, 
            all_str_logits_curr.transpose(0, 1), inputs, input_lens
        )
        all_sent_logprobs += logprobs_curr
        st = en

    if get_hidden_states:
        max_middle_dim = max(tensor.shape[1] for tensor in all_hidden_states)
        padded_tensors = [pad_tensor(tensor, max_middle_dim) for tensor in all_hidden_states]
        all_hidden_states = torch.cat(padded_tensors, dim=0)
    else:
        all_hidden_states = None

    return all_sent_logprobs, all_hidden_states
    
def compute_perplexity_from_logprobs(all_logprobs):
    """
    Compute perplexity from logprobs. works for both torch and numpy arrays.
    Also works if we want to marginalize parses
    """
    if type(all_logprobs[0]) == torch.Tensor:
        total_logprob = np.sum([torch.sum(p).item() for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    elif len(all_logprobs[0]) == 2:
        ### sent logprb, length
        total_len = np.sum([_len for logprob_set, _len in all_logprobs])
        total_logprob = np.sum(
            [logsumexp(logprob_set) for logprob_set, _len in all_logprobs]
        )
    else:
        total_logprob = np.sum([np.sum(p) for p in all_logprobs])
        total_len = np.sum([len(s) for s in all_logprobs])
    return np.exp(-total_logprob / total_len)

def get_parsing_accuracy(predicted_parses, gold_parses, split):
    """Compute parsing scores for predicted parses."""

    def get_brackets(parse):
        p_set = set()

        def get_brackets_helpers(t, st):
            if type(t) == str:
                return 1
            else:
                l1_len = get_brackets_helpers(t[0], st)
                l2_len = get_brackets_helpers(t[1], st + l1_len)
                p_set.add((st, st + l1_len + l2_len - 1))
                return l1_len + l2_len

        get_brackets_helpers(parse, 0)
        return p_set

    gold_brackets = [get_brackets(parse) for parse in gold_parses]
    pred_brackets = [get_brackets(parse) for parse in predicted_parses]

    def get_score(set_1, set_2):
        score = 0.0
        lengthwise = {}
        for p in set_2:
            if p in set_1:
                score += 1
                if (p[1] - p[0]) not in lengthwise:
                    lengthwise[p[1] - p[0]] = 0
                lengthwise[p[1] - p[0]] += 1
        return (score, lengthwise)
    
    def get_base_lengthwise_stats(brackets):
        lengthwise = {}
        for b in brackets:
            if (len(b) == 0):
                continue
            max_length = -1
            for p in b:
                if (p[1] - p[0]) not in lengthwise:
                    lengthwise[p[1] - p[0]] = 0
                max_length = max(max_length, p[1] - p[0])
                lengthwise[p[1] - p[0]] += 1
            # remove full sentence bracket
            lengthwise[max_length] -= 1
        return lengthwise

    precision = sum(
        [get_score(gold, pred)[0] for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    recall = sum(
        [get_score(pred, gold)[0] for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    bracket_len = sum(len(b) for b in pred_brackets) + 1e-9 # avoiding div by zero
    precision /= 1.0 * bracket_len
    recall /= 1.0 * bracket_len

    pred_lengthwise = {}
    for gold, pred in zip(gold_brackets, pred_brackets):
        _, lengthwise = get_score(gold, pred)
        if (len(lengthwise.keys()) == 0):
            continue
        max_length = max(lengthwise.keys())
        for key in lengthwise:
            if key not in pred_lengthwise:
                pred_lengthwise[key] = 0
            # remove full sentence bracket
            if key != max_length:
                pred_lengthwise[key] += lengthwise[key]

    gold_lengthwise = get_base_lengthwise_stats(gold_brackets)

    lengthwise_stats = {}
    for key in gold_lengthwise:
        if (gold_lengthwise[key] == 0):
            continue
        if (key not in pred_lengthwise):
            lengthwise_stats[key] = 0
        else:
            lengthwise_stats[key] = pred_lengthwise[key]/gold_lengthwise[key]

    paired_lists = []
    for key in lengthwise_stats:
        paired_lists.append((key, lengthwise_stats[key]))
    paired_lists.sort()

    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + 1e-10),
    }

def convert_tree_to_tuple_and_collate(tree, word_boundaries):
    """Convert NLTK tree to a tuple representation. Collapse subwords"""
    def fix(t):
        if type(t) == str:
            return t
        elif len(t) == 1:
            return fix(t[0])
        else:
            all_children = [c for c in t]
            span_subwords = t.leaves()
            one_word = True
            for idx in range(1, len(span_subwords)):
                if span_subwords[idx][0] == 'Ġ' or span_subwords[idx][0] == '▁':
                    one_word=False
                    break
            if one_word:
                return "".join(span_subwords)
            else:
                return (fix(all_children[0]), fix(tuple(all_children[1:])))

    return fix(tree)

def convert_tree_to_tuple(tree):
    """Convert NLTK tree to a tuple representation. Collapse subwords"""
    def fix(t):
        if type(t) == str:
            return t
        elif len(t) == 1:
            return fix(t[0])
        else:
            all_children = [c for c in t]
            return (fix(all_children[0]), fix(tuple(all_children[1:])))

    return fix(tree)