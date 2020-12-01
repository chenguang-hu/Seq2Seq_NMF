import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import time
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(BASE_DIR, "src")
sys.path.append(src)

from utils.util import load_data, load_de_vocab, load_en_vocab, pad_sequence, transformer_list, num2seq, load_pickle

from core.hyperparams import Hyperparams as hp

from model.seq2seq_nmf import Seq2Seq_NMF
from utils.data_loader import get_batch_data_flatten

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 20
LR = hp.lr  # 1e-4
lr_gamma = 0.5
save_dir = "save"
result_file = "result/pred.txt"

def train(train_iter, net, optimizer, vocab_size, pad, grad_clip=10, graph=False, debug=False, kl_mult=0):
    # choose nll_loss for training the objective function
    net.to(device)
    net.train()
    total_loss, batch_num = 0.0, 0
    criterion = nn.NLLLoss(ignore_index=pad)

    pbar = tqdm(train_iter)
    for idx, batch in enumerate(pbar):
        if graph:
            sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch

        # print(type(sbatch))
        tbatch = tbatch.to(device)
        turn_lengths = turn_lengths.to(device)

        batch_size = tbatch.shape[1]
        if batch_size == 1:
        # batchnorm will throw error when batch_size is 1
            continue

        optimizer.zero_grad()

        output = net(sbatch, tbatch, turn_lengths)


        if type(output) == tuple:
            # VHRED model, KL divergence add to the loss
            if len(output) == 2:
                output, kl_div = output
                bow_loss = None
            elif len(output) == 3:
                output, kl_div, bow_loss = output
            else:
                raise Exception('[!] wrong')
        else:
            kl_div, bow_loss = None, None
        loss = criterion(output[1:].view(-1, vocab_size),
                         tbatch[1:].contiguous().view(-1))
        if kl_div:
            loss += kl_mult * kl_div
        if bow_loss:
            loss += bow_loss
        
        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        batch_num += 1

        pbar.set_description(f'batch {batch_num}, training loss: {round(loss.item(), 4)}')

    # return avg loss
    return round(total_loss / batch_num, 4), kl_mult


def validation(data_iter, net, vocab_size, pad, graph=False, transformer_decode=False, debug=False):
    net.eval()
    batch_num, total_loss = 0, 0.0
    criterion = nn.NLLLoss(ignore_index=pad)

    pbar = tqdm(data_iter)

    for idx, batch in enumerate(pbar):
        if graph:
            sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch
        batch_size = tbatch.shape[1]
        if batch_size == 1:
            continue

        if graph:
            output = net(sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths)
        else:
            output = net(sbatch, tbatch, turn_lengths)

        if type(output) == tuple:
            # VHRED model, KL divergence add to the loss
            if len(output) == 2:
                output, _ = output
            elif len(output) == 3:
                output, _, _ = output
            else:
                raise Exception('[!] wrong')

        loss = criterion(output[1:].view(-1, vocab_size), tbatch[1:].contiguous().view(-1))

        total_loss += loss.item()
        batch_num += 1

        pbar.set_description(f'batch {idx}, dev loss: {round(loss.item(), 4)}')

    return round(total_loss / batch_num, 4)

src_w2idx, src_idx2w = load_de_vocab()
tgt_w2idx, tgt_idx2w = load_en_vocab()

DICT_NAME = 'news'
DICT_PATH = os.path.join("topic_model", DICT_NAME + '-nmf.npz')

topic_dict = torch.tensor(np.load(DICT_PATH)["dictionary"], dtype=torch.float).to(device)

net = Seq2Seq_NMF(len(src_w2idx), 256, len(tgt_w2idx), 512, 512, topics=topic_dict, topic_vocab_size=topic_dict.shape[1], teach_force=1, pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], utter_n_layer=2, pretrained=None)

optimizer = Adam(net.parameters(), lr=LR)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=hp.patience, verbose=True, cooldown=0, min_lr=1e-6)

pbar = tqdm(range(1, epochs + 1))
training_loss, validation_loss = [], []

for epoch in pbar:
    train_iter = get_batch_data_flatten(hp.source_train, hp.target_train, batch_size=128, maxlen=hp.maxlen, tgt_maxlen=25)
    val_iter = get_batch_data_flatten(hp.source_dev, hp.target_dev, batch_size=128, maxlen=hp.maxlen, tgt_maxlen=25)

    train_loss = train(train_iter, net, optimizer, len(tgt_w2idx), tgt_w2idx['<pad>'], grad_clip=3.0, kl_mult=0.0)

    with torch.no_grad():
        val_loss = validation(val_iter, net, len(tgt_w2idx), tgt_w2idx['<pad>'])

    state = {'net': net.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(save_dir, 'checkpoint_' + str(epoch) + '.pth'))

    pbar.set_description(f'Epoch: {epoch}, loss(train/dev): {train_loss}/{val_loss}, ppl(dev): {round(math.exp(val_loss), 4)}')


pbar.close()
print(f'[!] Done')