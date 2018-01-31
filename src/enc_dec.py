import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.autograd as ag
from torch.autograd import Variable
import torch.nn.functional as F

import lang
import data_proc as dp
import basic_rnn

MAX_PREDICTION_LENGTH = 20


class EncoderDecoderRNN(nn.Module):
    '''
        Inputs:
            -- A supervised translation batch, consisting of an input and an output batch.
            --{in,out}_vocab_size: The size of the {input, output} vocabulary. 
            --{in,out}_embedding_dim: The dimension of the {input, output} word vector embeddings.
            --hidden_dim: The dimension of the hidden layers in the Encoder.  The decoder hidden layers will have dimension n_directions*hidden_dim.
            --n_layer: The number of hidden layers.  See Graves (2014)
            -bidirectional: Whether the encoder is bidirectional.  The decoder will always be one-directional, but will have a hidden layer twice as large if bidirectional=True
            -pre_{src,tgt}_embedding: Pre-trained word embeddings for the source or target languages. 

        Outputs:
            --A variable holding the output  of the loss function (cross-entropy).
    '''

    def __init__(self,
                 in_vocab_size,
                 out_vocab_size,
                 in_embedding_dim,
                 out_embedding_dim,
                 hidden_dim,
                 n_layers=1,
                 bidirectional=False,
                 pre_src_embedding=None,
                 pre_tgt_embedding=None):
        super(EncoderDecoderRNN, self).__init__()
        self.out_vocab_size = out_vocab_size
        n_directions = 2 if bidirectional else 1
        self.encoder = basic_rnn.RNN(
            in_vocab_size,
            in_embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            pretrained_embedding=pre_src_embedding)
        self.decoder = basic_rnn.RNN(
            out_vocab_size,
            out_embedding_dim,
            hidden_dim,
            n_directions*n_layers,
            bidirectional=False,
            pretrained_embedding=pre_tgt_embedding)
        self.lin = torch.nn.Linear(
             hidden_dim, out_vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  #ignore padding

    def encode_decode(self, batch):
        batchsize = batch.perm.shape[0]
        _, code = self.encoder(batch.src)
        code = code[:, batch.perm, :]
        hidden_seq, _ = self.decoder(batch.tgt, code)
        padded_hidden_seq, _ = rnn.pad_packed_sequence(
            hidden_seq, batch_first=True
        )  #has dimension  batchsize *sequencelength *hidden_size
        out = self.lin(
            padded_hidden_seq
        )  #this has dimension  batchsize * sequenceslength *  out_vocab_size
        return out

    def forward(self, batch):
        out = self.encode_decode(batch)
        out = out.view(-1, self.out_vocab_size)
        goal = torch.cat(
            (batch.tgt.seqs[:, 1:],
             ag.Variable(
                 batch.tgt.seqs.data.new(batch.tgt.seqs.data.shape[0],
                                         1).fill_(0))),
            1)  #prediction is staggered.  at sequence element t we predict t+1
        out = self.loss(out, goal.view(-1))
        return out

    def predict(self, in_seq, in_seq_len=None):
        '''
            Args:
                -in_seq: A list of integers containing the sequence of interest.
                -in_seq_len: A list of one element, the length of the sequence in in_seq.  This allows the use of padding.
            Returns:
                - A list of integers containing the predicted output sequence.
        '''
        cuda = next(self.parameters()).is_cuda
        if in_seq_len == None:
            in_seq_len = [len(in_seq)]
        in_seq = dp.TranslationBatch(
            ag.Variable(torch.LongTensor(in_seq)).view(1, -1), in_seq_len)
        if cuda:
            in_seq = in_seq.cuda()
        _, hidden = self.encoder(in_seq)
        index = ag.Variable(torch.LongTensor([lang.SOS_TOKEN]).view(1, 1))
        if cuda:
            index = index.cuda()
        iter = 0
        indexes = [index.data[0][0]]
        while True:
            embedded = self.decoder.embed(index)
            out_vec, hidden = self.decoder.gru(embedded, hidden)
            weights = self.lin(out_vec).squeeze()
            _, index = torch.topk(weights, k=1)
            indexes.append(index.data[0])
            if index.data[0] == lang.EOS_TOKEN:
                break
            index = index.view(1, 1)
            iter = iter + 1
            if iter >= MAX_PREDICTION_LENGTH:
                indexes.append(lang.EOS_TOKEN)
                break
        return indexes
