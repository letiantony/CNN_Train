# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe


def read_examples(path, fields):
    es = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip('\n')
            if line == '':
                continue
            #text, label = line.split('\t')
            line_list = line.split('\t')
            label, text = line_list[0], "。".join(line_list[1:])
            es.append(data.Example.fromlist([text, label], fields))
    return es


def load_dataset(folder, batch_size, max_len, pretrained_embedding_path):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """

    #tokenize = lambda x: x.split()
    def tokenize(x): return [c for c in x]
    TEXT = data.Field(
        sequential=True,
        tokenize=tokenize,
        lower=True,
        fix_length=max_len)
    #TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(
        sequential=True,
        tokenize=lambda x: x.split(','),
        batch_first=True)
    #LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    fields = [('text', TEXT), ('label', LABEL)]
    train_examples = read_examples(folder + '/train.txt', fields)
    dev_examples = read_examples(folder + '/dev.txt', fields)
    test_examples = read_examples(folder + '/test.txt', fields)
    examples = train_examples + dev_examples + test_examples
    train_data = data.Dataset(train_examples, fields)
    test_data = data.Dataset(test_examples, fields)
    dev_data = data.Dataset(dev_examples, fields)
    all_data = data.Dataset(examples, fields)

    #train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    LABEL.build_vocab(all_data)
    print(LABEL.vocab)
    TEXT.build_vocab(
        all_data,
        vectors=Vectors(
            name=pretrained_embedding_path,
            cache='.'))
    #TEXT.build_vocab(all_data, vectors=Vectors(name='giga-kg-vector-100.bin.txt', cache='.'))

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    # train_data, dev_data = train_data.split() # Further splitting of
    # training_data to create new training_data & dev_data
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data), batch_size=batch_size, sort_key=lambda x: len(
            x.text), repeat=False, shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)
    label_size = len(LABEL.vocab)

    return TEXT, vocab_size, label_size, word_embeddings, train_iter, dev_iter, test_iter, LABEL, TEXT
