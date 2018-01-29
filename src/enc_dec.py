import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.autograd as ag
import torch.nn.functional as F

import lang
import data_proc as dp


class EncoderRNN(nn.Module):
    ''' 
        Args:
            -vocab_size: the size of the vocabulary the encoder is to encode.
            -embedding_dim: size of word vectors to use
            -hidden_dim: dimension of the hidden units
            -n_layers: number of layers in the recurrent neural net.  See Graves (2014)

        inputs:
            -batch: a TranslationBatch of input data
            -init_hidden:  Either a variable giving the initial value for the hidden state, or None.  None is appropriate in most cases.
        outputs:
            -final hidden units for each layer
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)

    def forward(self, batch, init_hidden=None):
        batch_size = batch.seqs.shape[0]
        embedded = self.embedding(batch.seqs)
        packed = rnn.pack_padded_sequence(
            embedded, batch.lengths, batch_first=True)
        if init_hidden is None:
            _, final_hidden = self.gru(packed)
        else:
            _, final_hidden = self.gru(packed, init_hidden)
        return final_hidden


class DecoderRNN(nn.Module):
    '''
        Args:
            -vocab_size: the size of the vocabulary the decoder is to decode
            -embedding_dim: dimension of word vectors to use
             -hidden_dim: dimension of hidden units
            -n_layers: number of layers in recurrent neural net.  See Graves (2014)
           
        inputs:
            -batch: a translation batch of correct output data
            -code: the hidden units output from the last layer of the encoder.  Should be 3d with dimensions n_layers by batch_size by hidden_dim
        outputs:
            -the predictions of the decoder at every time step (in packed form)

    '''

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)

    def forward(self, batch, code):
        embedded = self.embedding(batch.seqs)
        packed = rnn.pack_padded_sequence(
            embedded, batch.lengths, batch_first=True)
        hidden_seq, _ = self.gru(packed, code)
        return hidden_seq


MAX_PREDICTION_LENGTH = 20


class EncoderDecoderRNN(nn.Module):
    '''
        inputs:
            -- A supervised translation batch, conssiting of an input and an output batch.
        outputs:
            --variable holding the value of the loss function (cross-entropy )
    '''

    def __init__(self,
                 in_vocab_size,
                 out_vocab_size,
                 in_embedding_dim,
                 out_embedding_dim,
                 hidden_dim,
                 n_layers=1):
        super(EncoderDecoderRNN, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.encoder = EncoderRNN(in_vocab_size, in_embedding_dim, hidden_dim,
                                  n_layers)
        self.decoder = DecoderRNN(out_vocab_size, out_embedding_dim,
                                  hidden_dim, n_layers)
        self.lin = torch.nn.Linear(hidden_dim, out_vocab_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  #ignore padding

    def encode_decode(self, batch):
        batchsize = batch.perm.shape[0]
        code = self.encoder(batch.src)
        code = code[:, batch.perm, :]
        hidden_seq = self.decoder(batch.tgt, code)
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
             batch.tgt.seqs.new(batch.tgt.seqs.shape[0],1).fill_(0))  ,
            1)  #prediction is staggered.  at sequence element t we predict t+1
        out = self.loss(out, goal.view(-1))
        return out

    def predict(self, in_seq, in_seq_len=None):
        '''
            Args:
                -in_seq: list of intgers. the sequence of interest
                -in_seq_len: list of one element, the length of the sequence in in seq.  Allows padding
            Returns:
                - a list of integers. the predicted output sequence
        '''
        cuda = next(self.parameters()).is_cuda
        if in_seq_len == None:
            in_seq_len = [len(in_seq)]
        in_seq = dp.TranslationBatch(
            ag.Variable(torch.LongTensor(in_seq)).view(1, -1), in_seq_len)
        if cuda:
            in_seq = in_seq.cuda()
        hidden = self.encoder(in_seq)
        index = ag.Variable(torch.LongTensor([lang.SOS_TOKEN]).view(1, 1))
        if cuda:
            index = index.cuda()
        iter = 0
        indexes = [index.data[0][0]]
        while True:
            embedded = self.decoder.embedding(index)
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
