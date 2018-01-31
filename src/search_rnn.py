import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

import basic_rnn as basic
#plan: will likely have to process target batches one timestep at a time, since I need the hidden unit from the last time step to get the correct context vector for this timestep
#how to deal with differing sequence lengths?
#could just compute up to maximum sequence length and just throw away excess time steps.  Inefficent,  But might be less so than looping through sequences beause of python overhead
#or: could do the following: each time step, check which sequences have already terminated.  Create a new variable consisting of the sequences above that. copy any just-completed sequences to an output storage variable that starts out as zero.  Apply one step of the gru to the variable of still runnning sequences.
#Seems like it could work.
#either way, probably not a big deal because of the dominance of softmax in computation cost.

#problem: Raw softmax does not have an ignore_index argument, but this is what we need to compute the ALpha function batchwise.  Guess I could write one in CUDA if it was important to performance.
#but probably it is not worth it
#actually: here is a cool hack: use -infinity.
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
        
         
   def forward(self,batch):
      batchsize=batch.perm.shape[0]
      src_hidden_seq, =self.encoder(batch.src) 
      src_hidden_seq=rnn.pad_packed_sequence(src_hidden_seq)
      src_hidden_seq=src_hidden_seq[batch.perm, :,:] #has dimensions batchsize by max sequence length by src_hidden_dim*2
      Uh=self.U(src_hidden_seq) #has dimension batchsize by max sequence length by tgt_hidden_dim
      tgt_hidden_layer = Variable(torch.zeros(batchsize,self.decoder.n_layers,self.decoder.hidden_dim ))
      max_seq_len=src_hidden_seq.shape[1]
      for i in range(max_seq_len):
        #compute the eij for each j in the input sequence
        


class EFunc(nn.Module):


    #prob: how do we take the softmax of the output of this?
class AFunc(nn.Module):
    '''
        Input:
            --tgt_hidden_past: Variable with dimenisons batchsize by 1 by tgt_hidden_dim. Provides the value of the last hidden layer of the decoder in the previous timestep.
            --Uh: Variable with dimensions batchsize by max_sequence_length by tgt_hidden__dun
        Returns:
            -- Variable with dimension batchsize by max_seq_len by 1
    '''
    def __init__(self, tgt_hidden_dim):
       self.v=nn.Linear(tgt_hidden_dim,1)
       self.W=nn.Linear(tgt_hidden_dim,tgt_hidden_dim)
       self.tanh=nn.Tanh()
    def forward(self,tgt_hidden_past, Uh): 
        return self.v(self.tanh(self.W(tgt_hidden_past)+Uh )) 
