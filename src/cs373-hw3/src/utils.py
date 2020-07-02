
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import frequency_Counter

vocab = {}

def read_data(filepath, num_ex=-1):
    if num_ex == -1:
        num_ex = 1000
    train_data = open(filepath + 'train.tsv', 'r').read().split('\n')[1:num_ex+1]
    train_data = [line.split('\t') for line in train_data]
    tr_x, tr_y = [ex[1] for ex in train_data], [get_label(ex[2]) for ex in train_data]
    test_data = open(filepath + 'test.tsv', 'r').read().split('\n')[1:num_ex+1]
    test_data = [line.split('\t') for line in test_data]
    te_x, te_y = [ex[1] for ex in test_data], [get_label(ex[2]) for ex in test_data]
    return ((tr_x, tr_y), (te_x, te_y))


def get_feature_vectors(x, binary=False):
    #Add more features here, if you want to!
    global vocab
    v_size = len(vocab)
    if binary:
        X = [get_binary_features(transform_data(ex), vocab_size=v_size) for ex in x]
    else:
        X = [get_frequency_features(transform_data(ex), vocab_size=v_size) for ex in x]
    return X

    """
def build_vocab(filepath, vocab_size=10000):
    global vocab
    vocab_dict = {}
    data = open(filepath + 'train.tsv', 'r').read().split('\n')[1:]
    examples = [ex.split('\t') for ex in data]
    for example in examples:
        if len(example) == 3:
            tokens = example[1].strip().lower().split()
            for tok in tokens:
                if tok in vocab_dict:
                    vocab_dict[tok] += 1
                else:
                    vocab_dict[tok] = 1
    vocab_list = []
    for tok in vocab_dict:
        vocab_list.append((tok, vocab_dict[tok]))
    freq_sorted_vocab = sorted(vocab_list, key=lambda x:x[1], reverse=True)
    pruned_vocab = freq_sorted_vocab[:vocab_size-3]
    vocab['<BOS>'] = 0 #beginning of the post token
    vocab['<EOS>'] = 1 #end of the post token
    vocab['<oov>'] = 2 #out-of-vocabulary token
    with open(filepath + 'vocab', 'w') as outfile:
        for i, tup in enumerate(pruned_vocab):
            outfile.write(tup[0] + '\n')
            vocab[tup[0]] = i + 3
    """
"""
def buildVocab(filepath):
    global tokens
    file = open(filepath, "r", encoding="utf8", errors="namespace").read()
    tokens = file.split("\n")
"""

def transform_data(ex_str):
    global vocab
    ex_str = '<BOS> ' + ex_str.strip().lower() + ' <EOS>'
    ex_toks = []
    tokens = ex_str.split()
    for tok in tokens:
        if tok in vocab:
            ex_toks.append(vocab[tok])
        else:
            ex_toks.append(vocab['<oov>'])
    return ex_toks


def get_binary_features(ex_toks, vocab_size=10000):
    fv = [0.] * vocab_size
    for tok in ex_toks:
        assert(tok < vocab_size)
        fv[tok] = 1.
    return fv


def get_frequency_features(ex_toks, vocab_size=10000):
    fv = [0.] * vocab_size
    for tok in ex_toks:
        assert(tok < vocab_size)
        fv[tok] += 1.
    return fv

def get_label(l_str):
    if l_str.strip().lower() in {'positive', 'pos', 'y', 'yes', '1'}:
        return 1
    else:
        return -1

if __name__ == '__main__':
    fpath = filepath = '../data/given/'
    build_vocab(fpath)

