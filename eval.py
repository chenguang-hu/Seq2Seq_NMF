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
from utils.metric import cal_Distinct

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

loadFilename = "save/checkpoint_15.pth"
result_file = "result/pred.txt"

DICT_NAME = 'news'
DICT_PATH = os.path.join("topic_model", DICT_NAME + '-nmf.npz')

topic_dict = torch.tensor(np.load(DICT_PATH)["dictionary"], dtype=torch.float).to(device)

if loadFilename:
    checkpoint = torch.load(loadFilename)
    net_sd = checkpoint['net']

src_w2idx, src_idx2w = load_de_vocab()
tgt_w2idx, tgt_idx2w = load_en_vocab()

net = Seq2Seq_NMF(len(src_w2idx), 256, len(tgt_w2idx), 512, 512, topics=topic_dict, topic_vocab_size=topic_dict.shape[1], teach_force=1, pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'], utter_n_layer=2, pretrained=None)

net.load_state_dict(net_sd)

net.to(device)

def eval_distinct(pred_path):
    with open(pred_path) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            line = line.lower()    # lower the case
            if idx %4 == 1:
                line = line.replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx %4 == 2:
                line = line.replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())
    
    assert len(ref) == len(tgt)
    # Distinct-1, Distinct-2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    rdistinct_1, rdistinct_2 = cal_Distinct(references)

    return distinct_1, distinct_2


def translate(data_iter, net):
    net.eval()

    criterion = nn.NLLLoss(ignore_index=tgt_w2idx['<pad>'])
    total_loss, batch_num = 0.0, 0

    with open(result_file, 'w') as f:
        pbar = tqdm(data_iter)
        for batch in pbar:
            sbatch, tbatch, turn_lengths = batch

            batch_size = tbatch.shape[1]

            turn_size = len(sbatch)

            src_pad, tgt_pad = src_w2idx['<pad>'], tgt_w2idx['<pad>']
            src_eos, tgt_eos = src_w2idx['<eos>'], tgt_w2idx['<eos>']

            output, _ = net.predict(sbatch, len(tbatch), turn_lengths, loss=True)

            # print(output)

            with torch.no_grad():
                f_l = net(sbatch, tbatch, turn_lengths)
                if type(f_l) == tuple:
                    f_l = f_l[0]
        
            loss = criterion(f_l[1:].view(-1, len(tgt_w2idx)), tbatch[1:].contiguous().view(-1))

            batch_num += 1
            total_loss += loss.item()

            for i in range(batch_size):
                ref = list(map(int, tbatch[:, i].tolist()))
                tgt = list(map(int, output[:, i].tolist())) 

                # src = [sbatch[j][:, i].tolist() for j in range(turn_size)]
                src = list(map(int, sbatch[:, i].tolist()))

                # filte the <pad>
                ref_endx = ref.index(tgt_pad) if tgt_pad in ref else len(ref)
                ref_endx_ = ref.index(tgt_eos) if tgt_eos in ref else len(ref)
                ref_endx = min(ref_endx, ref_endx_)
                ref = ref[1:ref_endx]
                ref = ' '.join(num2seq(ref, tgt_idx2w))
                ref = ref.replace('<sos>', '').strip()

                tgt_endx = tgt.index(tgt_pad) if tgt_pad in tgt else len(tgt)
                tgt_endx_ = tgt.index(tgt_eos) if tgt_eos in tgt else len(tgt)
                tgt_endx = min(tgt_endx, tgt_endx_)
                tgt = tgt[1:tgt_endx]
                tgt = ' '.join(num2seq(tgt, tgt_idx2w))
                tgt = tgt.replace('<sos>', '').strip()

                # source = []
                # for item in src:
                #     item_endx = item.index(src_pad) if src_pad in item else len(item)
                #     item_endx_ = item.index(src_eos) if src_eos in item else len(item)
                #     item_endx = min(item_endx, item_endx_)
                #     item = item[1:item_endx]
                #     item = num2seq(item, src_idx2w)
                #     source.append(' '.join(item))
                # src = ' __eou__ '.join(source)
                src_endx = src.index(src_pad) if src_pad in src else len(src)
                src_endx_ = src.index(src_eos) if src_eos in src else len(src)
                sec_endx = min(src_endx, src_endx_)
                src = src[1:src_endx]
                src = ' '.join(num2seq(src, src_idx2w))

                f.write(f'- src: {src}\n')
                f.write(f'- ref: {ref}\n')
                f.write(f'- tgt: {tgt}\n\n')
    # print(total_loss)
    l = round(total_loss / batch_num, 4)
    print(f'[!] write the translate result into pred.txt')
    print(f'[!] test loss: {l}, test ppl: {round(math.exp(l), 4)}')

    return math.exp(l)


test_iter = get_batch_data_flatten(hp.source_test, hp.target_test, batch_size=128, maxlen=hp.maxlen, tgt_maxlen=25)

ppl = translate(test_iter, net)

print(f'ppl(test): {round(ppl, 4)}')

distinct_1, distinct_2 = eval_distinct("result/pred.txt")

print(f'distinct_1: {distinct_1}')
print(f'distinct_2: {distinct_2}')