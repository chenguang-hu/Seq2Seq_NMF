from __future__ import print_function
import numpy as np
import codecs
import os
import regex
import torch
from torch.utils.data import Dataset
import torch.utils.data
import torch.nn as nn
from collections import Counter
import pickle

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = os.path.join(BASE_DIR, "src")
sys.path.append(src)

# print(BASE_DIR)

from utils.util import *


def get_batch_data_flatten(src, tgt, batch_size, maxlen, tgt_maxlen):
    # flatten batch data for unHRED-based models (Seq2Seq)
    # return long context for predicting response
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_de_vocab()
    tgt_w2idx, tgt_idx2w = load_en_vocab()
    
    # [datasize, lengths], [datasize, lengths]
    src_dataset, tgt_dataset = load_data_flatten(src, tgt, maxlen, tgt_maxlen)

    turns = [len(i) for i in src_dataset]
    turnsidx = np.argsort(turns)
    
    # sort by the lengths
    src_dataset = [src_dataset[i] for i in turnsidx]
    tgt_dataset = [tgt_dataset[i] for i in turnsidx]

    # generate the batch
    turns = [len(i) for i in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        
        bs = len(sbatch)

        # pad sbatch and tbatch
        turn_lengths = [len(sbatch[i]) for i in range(bs)]
        pad_sequence(src_w2idx['<pad>'], sbatch, bs)
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # [seq_len, batch]
        sbatch = torch.tensor(sbatch, dtype=torch.long).transpose(0, 1)
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths