import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

import basic_rnn as basic
import data_proc as dp

class SearchRNN(nn.Module):
    '''
        Based on 'Neural Machine Translation by Jointly Learning to Align and Translate' by Bahdanau, Cho, and Bengio 
    '''

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_embedding_dim,
                 tgt_embedding_dim,
                 src_hidden_dim,
                 tgt_hidden_dim,
                 n_layers=1,
                 pre_src_embedding=None,
                 pre_tgt_embedding=None):
        super(SearchRNN, self).__init__()
        self.tgt_hidden_dim=tgt_hidden_dim
        self.src_hidden_dim=src_hidden_dim
        self.tgt_vocab_size=tgt_vocab_size
        self.encoder = basic.RNN(
            vocab_size=src_vocab_size,
            embedding_dim=src_embedding_dim,
            hidden_dim=src_hidden_dim,
            n_layers=n_layers,
            bidirectional=True,
            pretrained_embedding=pre_src_embedding)
        self.decoder = basic.RNN(
            vocab_size=tgt_vocab_size,
            embedding_dim=tgt_embedding_dim,
            hidden_dim=tgt_hidden_dim,
            n_layers=n_layers,
            bidirectional=False,
            pretrained_embedding=pre_tgt_embedding,
            extra_input_dim=2 * src_hidden_dim)
        self.U = nn.Linear(2 * src_hidden_dim, tgt_hidden_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  #ignore padding
        self.afunc = AFunc(tgt_hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.lin_out = nn.Linear(tgt_hidden_dim, tgt_vocab_size)

    def forward(self, batch):
        batchsize = batch.src.seqs.shape[0]
        src_max_seq_len = batch.src.seqs.shape[1]
        tgt_max_seq_len = batch.tgt.seqs.shape[1]

        #import pdb; pdb.set_trace()
        src_hidden_seq, _ = self.encoder(batch.src)
        src_hidden_seq,_ = rnn.pad_packed_sequence(src_hidden_seq, batch_first=True)
        src_hidden_seq = src_hidden_seq[batch.perm, :, :]  #has dimensions batchsize by src_max sequence length by src_hidden_dim*2
        Uh = self.U(
            src_hidden_seq
        )  #has dimension batchsize by src_max_seq_length by tgt_hidden_dim
        cur_tgt_hidden_layer = Variable(
            torch.zeros(batchsize, self.decoder.n_layers,
                        self.decoder.hidden_dim))

        padding_knockout = Variable(src_hidden_seq.data.new(batchsize, src_max_seq_len,
                                              1).zero_())
        for k in range(batchsize):
            if batch.src.lengths[k]<src_max_seq_len:
                padding_knockout[k, batch.src.lengths[k]:, 0] = -float("inf")

        num_continuing = batchsize  # at the current timestep, the number of setences that have yet to terminate
        decoder_output = Variable(
            torch.zeros(batchsize, tgt_max_seq_len, self.tgt_hidden_dim))
        for i in range(tgt_max_seq_len):
            while batch.tgt.lengths[num_continuing - 1] <= i:
                num_continuing -= 1
            assert (num_continuing > 0)

            e_batch = self.afunc(cur_tgt_hidden_layer[:num_continuing, :, :], Uh[:num_continuing,:,:])
            #import pdb; pdb.set_trace()
            e_batch += padding_knockout[:num_continuing, :, :]
            alpha_batch = self.softmax(e_batch)
            
            c_batch = torch.sum(
                alpha_batch * src_hidden_seq[:num_continuing, :, :],
                dim=1
            )  #multiplication here is pointwise and broadcast. #result should have dimensions num_continuing by 2*src_hidden_dim
            c_batch=c_batch.view(num_continuing,1,2*self.src_hidden_dim)
            
            cur_tgt =batch.tgt.first_k_at_t(k=num_continuing, t=i)

            hidden_out, cur_tgt_hidden_layer = self.decoder(
                cur_tgt, cur_tgt_hidden_layer, extra_input=c_batch)
            cur_tgt_hidden_layer= cur_tgt_hidden_layer.transpose(0,1) #batchsize needs to come first so we can apply AFunc
            
            hidden_out, _ = rnn.pad_packed_sequence(hidden_out, batch_first=True)  
            #hidden_out should have dimensions num_continuing by 1 by tgt_hidden_dim
            
            decoder_output[:num_continuing, i, :] = hidden_out
        out = self.lin_out(decoder_output)  
        # out should have dimension batch_size by tgt_max_seq_len by tgt_vocab_size
        #import pdb; pdb.set_trace()
        goal = torch.cat(
            (batch.tgt.seqs[:, 1:],
             Variable(batch.tgt.seqs.data.new(batchsize, 1).fill_(0))),
            1)  #prediction is staggered.  at sequence element t we predict t+1
        return self.loss(out.view(-1,self.tgt_vocab_size), goal.view(-1))


class AFunc(nn.Module):
    '''
        Input:
            --tgt_hidden_past: Variable with dimenisons num_continuing by 1 by tgt_hidden_dim. Provides the value for batches in range(num_continuing)  of the last hidden layer of the decoder in the previous timestep (i.e. timestep i-1 in the paper).
            --Uh: Variable with dimensions num_continuing by src_max_sequence_length by tgt_hidden_dim
        Returns:
            -- Variable with dimension num_continuing by src_max_seq_len by 1.  Its (k,j) entry is the value of e_{ij} for the kth batch.  See pg 3 in  Bahdanau et al. 
    '''

    def __init__(self, tgt_hidden_dim):
        super(AFunc, self).__init__()
        self.v = nn.Linear(tgt_hidden_dim, 1)
        self.W = nn.Linear(tgt_hidden_dim, tgt_hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, tgt_hidden_past, Uh):
        return self.v(self.tanh(self.W(tgt_hidden_past) + Uh))
