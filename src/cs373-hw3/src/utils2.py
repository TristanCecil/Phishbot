vocab = {}

def get_feature_vectors(x, binary=False):
    #Add more features here, if you want to!
    global vocab
    v_size = len(vocab)
    if binary:
        X = [get_binary_features(transform_data(ex), vocab_size=v_size) for ex in x]
    else:
        X = [get_frequency_features(transform_data(ex), vocab_size=v_size) for ex in x]
    return X



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