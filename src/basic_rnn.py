import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.autograd as ag
from torch.autograd import Variable
import torch.nn.functional as F

import lang
import data_proc as dp


def fill_missing_embeddings(seqs, missing_vecs, missing_dict, dimension):
    out = Variable(torch.zeros(seqs.shape[0], seqs.shape[1], dimension))
    for i in range(seqs.shape[0]):
        for j in range(seq.shape[1]):
            if seqs.data[i][j] in missing_dict:
                out[i, j, :] = out[i, j, :] + missing[missing_dict[seqs.data[i]
                                                                   [j]], :]
    return out


class RNN(nn.Module):
    '''
        Args:
            -vocab_size: the size of the vocabulary of input vectors. 
            -embedding_dim: dimension of word vectors to use
            -hidden_dim: dimension of hidden units
            -n_layers: number of layers in recurrent neural net.  See Graves (2014)
           -bidirectional: whether to use a bidirectional RNN 
           -extra_input_dim: The dimension of the extra input vectors, which are received as an additional piece of input and concatenated to the word embeddings.   
           -pack: whether to pack the sequence before feeding it to the neural network
        inputs:
            -batch: a translation batch of input data 
            -code: initial hidden units.  If this RNN is a decoder, these are the hidden units  from the last layer of the encoder.  Should be 3d with dimensions n_layers by batch_size by hidden_dim
            -extra_input: should have dimension batchsize by seqlength by extra_input_dim
        outputs:
            -the predictions at every time step in packed form (main output of decoder RNNs)
            -final hidden units of each layer  (main output of encoder RNNs). Has dimensions (num_layers * num_directions, batch, hidden_size)
    '''

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 n_layers=1,
                 bidirectional=False,
                 pretrained_embedding=None,
                 extra_input_dim=0, pack=True):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.extra_input_dim = extra_input_dim
        self.pack=pack
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim + extra_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.n_directions = 2 if bidirectional else 1
        if pretrained_embedding is not None:
            self.pretrained = True
            self.embedding.weights = pretrained_embedding.weights
            self.missing = pretrained_embedding.missing
            self.missing_dict = pretrained_embedding.missing_dict
        else:
            self.pretrained = False

    def embed(self, seqs):
        embedded = self.embedding(seqs)

        if self.pretrained:
            dimension = self.embedding.weight.shape[1]
            embedded = embedded + fill_missing_embeddings(
                seqs, self.missing, self.missing_dict, dimension)
        return embedded

    def forward(self, batch, code=None, extra_input=None):
        embedded = self.embed(batch.seqs)
        if self.extra_input_dim > 0:
            embedded = torch.cat((embedded, extra_input), 2)
        if self.pack:
            sequence = rnn.pack_padded_sequence(embedded, batch.lengths, batch_first=True)
        else:
            sequence=embedded
        if code is not None:
            hidden_seq, final_hidden = self.gru(sequence, code)
        else:
            hidden_seq, final_hidden = self.gru(sequence)
        #hidden_seq is packed sequence.   has dimension batch_size by (max) seq_length by hidden_dim*num_directions
        return hidden_seq, final_hidden
