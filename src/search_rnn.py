import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

import basic_rnn as basic
#todo: finish loop body. Add a paramter enabling the creation of RNNs that work entirely on unpacked sequences. Need the decoder to work that way here
#Also add the ability to extract a certain number of timesteps and batches from a supervised translation dataset.
class SearchRNN(nn.Module):
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
        self.encoder = basic.RNN(
            vocab_size= src_vocab_size,
            embedding_dim=src_embedding_dim,
            hidden_dim=src_hidden_dim,
            n_layers=n_layres,
            bidirectional=True,
            pretrained_embedding=pre_src_embedding)
       self.decoder = basic.RNN(
            vocab_size= tgt_vocab_size,
            embedding_dim=tgt_embedding_dim,
            hidden_dim=tgt_hidden_dim,
            n_layers=n_layers,
            bidirectional=False,
            pretrained_embedding=pre_tgt_embedding,
            extra_input_dim=2*src_hidden_dim)
        self.U=nn.Linear(2*src_hidden_dim,tgt_hidden_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  #ignore padding
        self.afunc=Afunc(tgt_hidden_dim) 
        self.softmax=nn.softmax(dim=1)

   def forward(self,batch):
      batchsize=batch.perm.shape[0]
      src_hidden_seq,_ =self.encoder(batch.src) 
      src_hidden_seq=rnn.pad_packed_sequence(src_hidden_seq)
      src_hidden_seq=src_hidden_seq[batch.perm, :,:] #has dimensions batchsize by (src) max sequence length by src_hidden_dim*2
      Uh=self.U(src_hidden_seq) #has dimension batchsize by (src) max sequence length by tgt_hidden_dim
      tgt_hidden_layer = Variable(torch.zeros(batchsize,self.decoder.n_layers,self.decoder.hidden_dim ))
      src_max_seq_len=src_hidden_seq.shape[1]
      padding_knockout=src_hidden_seq.new(bathsize,src_max_seq_len).zero_()
      num_continuing=batchsize # at the current timestep, how many setences have yet to terminate
      for k in range(batchsize):
          padding_knockout[batch.src.lengths[k]: ]=-float("inf")
      for i in range(max_seq_len):
          e_batch= self.afunc(tgt_hidden_layer[:num_continuing,:,:] ,Uh)
          e_batch+=padding_knockout[:num_continuing,:]
          alpha_batch=self.softmax(e_batch)
          c_batch=torch.sum(alpha_batch*src_hidden_seq[:num_continuing,:,:],dim=1) #result should have dimensions num_continuing by 2*src_hidden_dim 
          self.decoder( )


class AFunc(nn.Module):
    '''
        Input:
            --tgt_hidden_past: Variable with dimenisons batchsize by 1 by tgt_hidden_dim. Provides the value of the last hidden layer of the decoder in the previous timestep (i.e. timestep i-1 in the paper).
            --Uh: Variable with dimensions batchsize by max_sequence_length by tgt_hidden_dim
        Returns:
            -- Variable with dimension batchsize by src_max_seq_len.  Its (k,j) entry is the value of e_{ij} for the kth batch.  See pg 3 in  Bahdanau et al. 
    '''
    def __init__(self, tgt_hidden_dim):
       self.v=nn.Linear(tgt_hidden_dim,1)
       self.W=nn.Linear(tgt_hidden_dim,tgt_hidden_dim)
       self.tanh=nn.Tanh()
    def forward(self,tgt_hidden_past, Uh): 
        return self.v(self.tanh(self.W(tgt_hidden_past)+Uh )).squeeze()
