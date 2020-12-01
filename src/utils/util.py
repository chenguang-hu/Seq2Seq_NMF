import numpy as np
import argparse
from collections import Counter
import pickle
import codecs
import os
import re
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from itertools import chain
import nltk
import math
import sys
nltk.download('punkt')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = os.path.join(BASE_DIR, "src")
sys.path.append(src)

from core.hyperparams import Hyperparams as hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    '''
    ('<pad>', 0), ('<unk>', 1), ('<sos>', 2), ('<eos>', 3), ('.', 4), ('__eou__', 5), (',', 6), ('i', 7), ('you', 8), ('?', 9)
    '''
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def num2seq(src, idx2w):
    # number to word sequence, src: [maxlen]
    return [idx2w[int(i)] for i in src]

def transformer_list(obj):
    # transformer [batch, turns, lengths] into [turns, batch, lengths]
    # turns are all the same for each batch
    turns = []
    batch_size, turn_size = len(obj), len(obj[0])
    for i in range(turn_size):
        turns.append([obj[j][i] for j in range(batch_size)])    # [batch, lengths]
    return turns


def pad_sequence(pad, batch, bs):
    maxlen = max([len(batch[i]) for i in range(bs)])
    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))

def clean(s):
    # this pattern are defined for cleaning the dailydialog dataset
    s = s.strip().lower()
    # s = re.sub(r'(\w+)\.(\w+)', r'\1 . \2', s)
    # s = re.sub(r'(\w+)-(\w+)', r'\1 \2', s)
    # s = re.sub(r'[0-9]+(\.[0-9]+)?', r'1', s)
    # s = s.replace('。', '.')
    # s = s.replace(';', ',')
    # s = s.replace('...', ',')
    # s = s.replace(' p . m . ', ' pm ')
    # s = s.replace(' P . m . ', ' pm ')
    # s = s.replace(' a . m . ', ' am ')
 
    return s

def pad_sequences(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence




def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequences(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

class PositionEmbedding(nn.Module):

    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)   # not the parameters of the Module

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# load data function for hierarchical models
# def load_data(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=True):
def load_data(src, tgt, maxlen, tgt_maxlen):
     
    src_w2idx, src_idx2w = load_de_vocab()
    tgt_w2idx, tgt_idx2w = load_en_vocab()
    # src_user, tgt_user = [], []
    # user_vocab = ['user0', 'user1']
    
    # src 
    with open(src) as f:
        src_dataset = []
        for line in tqdm(f.readlines()):
            # 这里将单词转换成小写
            line = clean(line)
            utterances = line.split('__eou__')    # only for chinese (zh50)

            # print(utterances)
            turn = []
            for utterance in utterances:
                if len(utterance.split()) == 0:
                    continue
                
                line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in utterance.split()] + [src_w2idx['<eos>']]
                # print(line)
                if len(line) > maxlen:
                    line = [src_w2idx['<sos>'], line[1]] + line[-maxlen:]
                turn.append(line)
                
            # print(turn)
            src_dataset.append(turn)
            
        '''
        context测试数据(2条)：
            Can you study with the radio on ? __eou__
            Can you study with the radio on ? __eou__ No , I listen to background music . __eou__

        经过处理后src_dataset：   
            [[[2, 28, 8, 497, 40, 10, 1690, 36, 9, 3]], [[2, 28, 8, 497, 40, 10, 1690, 36, 9, 3], [2, 57, 6, 7, 624, 11, 1274, 359, 4, 3]]]
        '''
        # print(src_dataset)

    # tgt
    with open(tgt) as f:
        tgt_dataset = []
        for line in tqdm(f.readlines()):
            line = clean(line)
            utterances = line.split('__eou__')
            # print(utterances)
            line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in utterances[0].split()] + [tgt_w2idx['<eos>']]
            if len(line) > maxlen:
                line = line[:tgt_maxlen] + [tgt_w2idx['<eos>']]
            tgt_dataset.append(line)
            
        '''
        response测试数据(2条)：
            No , I listen to background music . __eou__
            What is the difference ? __eou__

        经过处理后tgt_dataset:
            [[2, 49, 7, 6, 628, 11, 1657, 404, 4, 3], [2, 25, 16, 9, 839, 10, 3]] 
        '''
        # print(tgt_dataset)

    return src_dataset, tgt_dataset


def load_data_flatten(src, tgt, maxlen, tgt_maxlen):
    '''
    Used by vanilla seq2seq with attention and transformer
    '''
    # check the file, exist -> ignore
    src_prepath = os.path.splitext(src)[0] + '-flatten.pkl'
    tgt_prepath = os.path.splitext(tgt)[0] + '-flatten.pkl'
    if os.path.exists(src_prepath) and os.path.exists(tgt_prepath):
        print(f'[!] preprocessed file {src_prepath} exist, load directly')
        print(f'[!] preprocessed file {tgt_prepath} exist, load directly')
        with open(src_prepath, 'rb') as f:
            src_dataset = pickle.load(f)
        with open(tgt_prepath, 'rb') as f:
            tgt_dataset = pickle.load(f)
        return src_dataset, tgt_dataset
    else:
        print(f'[!] cannot find the preprocessed file')
    
    # sort by the lengths
    # src_w2idx, src_idx2w = load_pickle(src_vocab)
    # tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src_w2idx, src_idx2w = load_de_vocab()
    tgt_w2idx, tgt_idx2w = load_en_vocab()

    # sub function
    def load_(filename, w2idx, src=True):
        with open(filename) as f:
            dataset = []
            for line in tqdm(f.readlines()):
                line = clean(line)
                # if '<user0>' in line: user_c = '<user0>'
                # elif '<user1>' in line: user_c = '<user1>'
                line = line.replace('<user0>', 'user0')
                line = line.replace('<user1>', 'user1')
                line = [w2idx['<sos>']] + [w2idx.get(w, w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [w2idx['<eos>']]
                if src and len(line) > maxlen:
                    line = [w2idx['<sos>']] + line[-maxlen:]
                elif src == False and len(line) > tgt_maxlen:
                    line = line[:tgt_maxlen] + [w2idx['<eos>']]
                dataset.append(line)
        return dataset

    src_dataset = load_(src, src_w2idx, src=True)    # [datasize, lengths]
    tgt_dataset = load_(tgt, tgt_w2idx, src=False)    # [datasize, lengths]
    print(f'[!] load dataset over, write into file {src_prepath} and {tgt_prepath}')
    
    with open(src_prepath, 'wb') as f:
        pickle.dump(src_dataset, f)
    with open(tgt_prepath, 'wb') as f:
        pickle.dump(tgt_dataset, f)

    return src_dataset, tgt_dataset


src_w2idx, src_idx2w = load_de_vocab()
tgt_w2idx, tgt_idx2w = load_en_vocab()