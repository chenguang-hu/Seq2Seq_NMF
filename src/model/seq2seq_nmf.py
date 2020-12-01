import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import random
import numpy as np
import pickle
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = os.path.join(BASE_DIR, "src")
sys.path.append(src)

from model.layers import *
from utils.util import src_idx2w

def calculate_codes(topic_for_code, input_seq_for_code, voc, feature_path, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size = 64 for training, 1 for chatting
    nmfdict = np.load(feature_path)["feature_names"]
    # len(nmfdict) = 1000
    # new_input_seq = [batch_size, 1000]
    new_input_seq = torch.zeros(batch_size, len(nmfdict))

    '''
        new_input_seq is a matrix , all value in new_input_set are set to 0, shape: [batch, 1000],
        each line of new_input_seq match 1,000 topic words.

        input_seq_for_code is the transpose of the original matrix, the origin: [max_len, batch], after transpose: [batch, max_len]
        each line in input_seq_for_code denote a utterance, if a word in the utterance also in the list of topic word, 
        then the value of the corresponding position in new_input_seq will set to 1.

        Finally, new_input_seq will be a matrix with 0 & 1.
    '''
    # topic_for_code = [10, 1000]
    for i in range(batch_size):
        # input_seq_for_code = [batch_size, max_length]
        for j in range(len(input_seq_for_code[i])):
            input_seq_words = voc[input_seq_for_code[i][j].item()]
            for check_index in range(len(nmfdict)):
                if nmfdict[check_index] == input_seq_words:
                    new_input_seq[i][check_index] = 1

    # three_d_topic = [batch_size, 10, 1000]
    three_d_topic = topic_for_code.repeat(batch_size, 1, 1).to(device)
    # three_d_q = [batch_size, 1000, 1]
    three_d_q = new_input_seq.repeat(1, 1, 1).permute(1, 2, 0).to(device)

    # torch.bmm(three_d_topic, three_d_q) = [batch_size, 10, 1]
    return torch.bmm(three_d_topic, three_d_q)


class Encoder(nn.Module):
    
    def __init__(self, input_size, embed_size, hidden_size, topics, n_layers=1, dropout=0.5, pretrained=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.embed_size = embed_size
        self.n_layer = n_layers
        self.topics = topics
        
        self.embed = nn.Embedding(self.input_size, self.embed_size)
        # self.input_dropout = nn.Dropout(p=dropout)
        
        self.rnn = nn.GRU(embed_size, 
                          hidden_size, 
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

        # self.hidden_proj = nn.Linear(2 * n_layers * hidden_size, hidden_size)
        # self.bn = nn.BatchNorm1d(num_features=hidden_size)
            
        self.init_weight()
            
    def init_weight(self):
        # orthogonal init
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, src, inpt_lengths, hidden=None):

        batch_size = src.shape[1]

        # topic_for_code = [10, 1000]
        topic_for_code = self.topics
        
        input_seq_for_code = src.transpose(0, 1)

        feature_path = os.path.join("topic_model", 'news-nmf.npz')

        # codes = [batch_size, 10, 1]
        codes = calculate_codes(topic_for_code, input_seq_for_code, src_idx2w, feature_path, batch_size)

        # src: [seq, batch]
        embedded = self.embed(src)    # [seq, batch, embed]
        # embedded = self.input_dropout(embedded)

        if not hidden:
            hidden = torch.randn(2 * self.n_layer, src.shape[-1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
        
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, 
                                                     enforce_sorted=False)
        # hidden: [2 * n_layer, batch, hidden]
        # output: [seq_len, batch, 2 * hidden_size]
        output, hidden = self.rnn(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # fix output shape
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # hidden = hidden.sum(axis=0)    # [batch, hidden]
        
        # fix hidden
        # hidden = hidden.permute(1, 0, 2)
        # hidden = hidden.reshape(hidden.shape[0], -1)
        # hidden = self.bn(hidden)    # [batch, *]
        # hidden = self.hidden_proj(hidden)
        hidden = torch.tanh(hidden)
        
        # [seq_len, batch, hidden_size], [batch, hidden]
        return output, hidden, codes


class Decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, output_size, topics, topic_vocab_size, n_layers=2, dropout=0.5, pretrained=None):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(output_size, embed_size)

        # context attention
        self.attention = Attention(hidden_size) 

        # topic attention
        self.topic_attention = TopicAttention(topic_vocab_size, hidden_size, hidden_size)

        self.rnn = nn.GRU(embed_size, 
                          hidden_size,
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout))

        self.concat = nn.Linear(hidden_size * 2 + topic_vocab_size + 1000, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.topics = topics
        self.topic_vocab_size = topic_vocab_size
        
        self.init_weight()
        
    def init_weight(self):
        # orthogonal inittor
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, last_hidden, encoder_outputs, codes):
        # inpt: [batch]
        # last_hidden: [2, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch, embed_size]

        batch_size = last_hidden.shape[1]

        rnn_output, hidden = self.rnn(embedded, last_hidden)
        
        # attn_weights: [batch, 1, timestep of encoder_outputs]
        key = last_hidden.sum(dim=0)
        attn_weights = self.attention(key, encoder_outputs)
        
        # topic_attn_weights: [batch_size, 1, num_topics]
        topic_attn_weights = self.topic_attention(last_hidden.sum(dim=0), self.topics, encoder_outputs[-1])
            
        # context: [batch, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        topic_context = torch.bmm(topic_attn_weights, self.topics.repeat(batch_size, 1, 1))

        # topic_for_Pk = [batch_size, topic_vocab_size, num_topics]
        topic_for_Pk = self.topics.repeat(batch_size,1,1).permute(0,2,1)

        # codes = [batch_size, 10, 1]
        # Pk_context = [batch_size, topic_vocab_size, 1]
        Pk_context = torch.bmm(topic_for_Pk, codes)

        # rnn_output = [batch_size, hidden_size]
        rnn_output = rnn_output.squeeze(0)

        # context = [batch_size, hidden_size]
        context = context.squeeze(0)
        
        # topic_context = [batch_size, topic_vocab_size]
        topic_context = topic_context.squeeze(1)

        # Pk_context = [batch_size, topic_vocab_size]
        Pk_context = Pk_context.squeeze(2)

        # concat_input = [batch_size, hidden_size + hidden_size + topic_vocab_size + 1000]
        concat_input = torch.cat((rnn_output, context, topic_context, Pk_context), 1)

        # concat_output = [batch_size, hidden_size]
        concat_output = torch.tanh(self.concat(concat_input))

        # output = [batch_size, output_size]
        output = self.out(concat_output)

        output = F.log_softmax(output, dim=1)
        
        # output: [batch, output_size]
        # hidden: [2, batch, hidden_size]
        # hidden = hidden.squeeze(0)
        return output, hidden

class Seq2Seq_NMF(nn.Module):
    
    '''
    Compose the Encoder and Decoder into the Seq2Seq_NMF model
    '''
    
    def __init__(self, input_size, embed_size, output_size, 
                 utter_hidden, decoder_hidden, topics, topic_vocab_size, 
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5, 
                 utter_n_layer=1, src_vocab=None, tgt_vocab=None,
                 pretrained=None):
        super(Seq2Seq_NMF, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, topics,
                               n_layers=utter_n_layer, 
                               dropout=dropout,
                               pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, 
                               output_size, topics, topic_vocab_size, n_layers=utter_n_layer,
                               dropout=dropout,
                               pretrained=pretrained)
        self.teach_force = teach_force
        self.utter_n_layer = utter_n_layer
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        
    def forward(self, src, tgt, lengths):
        # src: [lengths, batch], tgt: [lengths, batch], lengths: [batch]
        # ipdb.set_trace()
        batch_size, max_len = src.shape[1], tgt.shape[0]
        
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        # encoder_output: [seq_len, batch, hidden_size]
        # hidden: [1, batch, hidden_size]
        encoder_output, hidden, codes = self.encoder(src, lengths)
        hidden = hidden[-self.utter_n_layer:]
        output = tgt[0, :]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output, codes)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output, codes)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        
        # output: [max_len, batch, output_size]
        return outputs
    
    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()

            encoder_output, hidden, codes = self.encoder(src, lengths)
            hidden = hidden[-self.utter_n_layer:]
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, encoder_output, codes)
                floss[t] = output
                # output = torch.max(output, 1)[1]    # [1]
                output = output.topk(1)[1].squeeze()
                outputs[t] = output    # output: [1, output_size]

            if loss:
                return outputs, floss
            else:
                return outputs 


if __name__ == "__main__":
    pass