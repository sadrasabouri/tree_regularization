from nltk import Tree

def reformat_tree(tree, in_vocab, is_first, gpt2 = False):
    # "retokenizes" tree according to given vocabulary
    
    single_word = True
    for idx, word in enumerate(tree.leaves()):
        if (idx == 0):
            continue
        if (word[0] == 'Ġ'):
            single_word = False
            break
    
    if single_word:
        # this needs processing, subwords 
        recreated = "".join(tree.leaves())
        if not is_first and recreated.startswith('Ġ'):
            # remove first space
            recreated = recreated[1:]
        if gpt2 and not is_first:
            tokens = in_vocab.tokenize(' ' + recreated)
        else:
            tokens = in_vocab.tokenize(recreated)

        if len(tokens) == 1:
            return Tree(tree.label(), [tokens[0]])
        else:
            return Tree(tree.label(), [Tree(tree.label(), [_]) for _ in tokens])

    return Tree(
        tree.label(),
        [reformat_tree(t, in_vocab, is_first = is_first and idx == 0, gpt2=gpt2) for idx, t in enumerate(tree)]
    )

def reformat_tree_generic(tree, in_vocab, separator = "", hf=False):
    # "retokenizes" tree according to given vocabulary
    # these are NLTK constituency trees, each leaf is a word
    
    if len(tree.leaves()) == 1:
        word = separator.join(tree.leaves())
        if hf:
            tokens = in_vocab.tokenize(word)
        else:
            tokens = [in_vocab.from_index[_] for _ in in_vocab(word)]

        if len(tokens) == 1:
            return Tree(tree.label(), [tokens[0]])
        else:
            return Tree(tree.label(), [Tree(tree.label(), [_]) for _ in tokens])

    return Tree(
        tree.label(),
        [reformat_tree_generic(t, in_vocab) for t in tree]
    )

def flatten(parse, add_eos, clean=False, separator = " "):
    def helper(p):
        if type(p) == str or type(p) == int:
            return p
        else:
            return separator.join(helper(x) for x in p)

    if type(parse) == Tree:
        words = separator.join(parse.leaves())
    else:
        words = helper(parse)

    if clean:
        # for HF tokenization, sentences are simply sequences of words without sow prefix
        words = words.split(separator)
        cleaned_words = []
        curr_word = ""
        for idx in range(len(words)):
            # TODO: this will be a sow token
            if (words[idx][0] == 'Ġ' or words[idx][0] == '▁'):
                if idx != 0:
                    cleaned_words.append(curr_word)
                curr_word = words[idx][1:]
            else:
                curr_word += words[idx]
        cleaned_words.append(curr_word)
        words = separator.join(cleaned_words)

    if add_eos:
        return "{} <eos>".format(words)
    else:
        return words
    
def binarize_tree(parse):
    if type(parse) == str:
        return parse
    else:
        if len(parse) == 1:
            return binarize_tree(parse[0])
        else:
            return (binarize_tree(parse[0]), binarize_tree(parse[1:]))
        
def get_word_boundaries(sentence, in_vocab, hf=False, gpt2=False):
    words = sentence.split(" ")
    boundaries = []
    for idx, word in enumerate(words):
        if gpt2 and idx != 0:
            word = " " + word
        if hf:
            tokens = in_vocab.tokenize(word)
        else:
            tokens = in_vocab(word)
        boundaries += [1] + [0]*(len(tokens)-1)
    return boundaries

def get_word_boundaries_sow(sentence):
    words = sentence.split(" ")
    boundaries = [1]
    for word in words[1:]:
        if word.startswith('Ġ'):
            boundaries.append(1)
        else:
            boundaries.append(0)
    return boundaries

def tree_to_parse_decisions(parse, start, parse_dict, check=True):
    # Builds a dict containing index of gold split for all gold subtrees
    # Accumulates results in parse_dict
    if (len(parse.leaves()) <= 2):
        # There is only one possible split for phrases of length 2
        return len(parse.leaves())
    
    # We know that subwords are never split between constituents. 
    # If all tokens after the first one in the current span do not start with 'G.', it should not be split further (single word)
    if check:
        single_word = True
        for idx, word in enumerate(parse.leaves()):
            if (idx == 0):
                continue
            if (word[0] == 'Ġ' or word[0] == '▁'):
                single_word = False
                break
        
        if single_word:
            return len(parse.leaves())
    
    # print(parse.leaves())

    while len(parse) == 1:
        parse = parse[0]

    s1 = len(parse[0].leaves())
    s2 = len(parse[1].leaves()) 

    tree_to_parse_decisions(parse[0], start, parse_dict, check)
    tree_to_parse_decisions(parse[1], start + s1, parse_dict, check)

    parse_dict[str(start) + ' ' + str(start + s1 + s2)] = start + s1

    return s1 + s2