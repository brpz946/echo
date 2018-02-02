import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

import basic_rnn as basic
import data_proc as dp
import lang

MAX_PREDICTION_LENGTH = 20
class SearchRNN(nn.Module):
    '''
        Based on 'Neural Machine Translation by Jointly Learning to Align and Translate' by Bahdanau, Cho, and Bengio 
    '''
    @staticmethod
    def construct(**args):
        if "hidden_dim" in args:
            args["src_hidden_dim"]=args["hidden_dim"]
            args["tgt_hidden_dim"]=args["hidden_dim"]
            del args["hidden_dim"]
        return SearchRNN(**args)

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
        self.U = nn.Linear(2 * src_hidden_dim, n_layers*tgt_hidden_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)  #ignore padding
        self.afunc = AFunc(tgt_hidden_dim,n_layers)
        self.softmax = nn.Softmax(dim=1)
        self.lin_out = nn.Linear(tgt_hidden_dim, tgt_vocab_size)

    def advance(self,num_continuing, cur_tgt_hidden_layer,src_hidden_seq, Uh,cur_tgt,padding_knockout=None):
            e_batch = self.afunc(cur_tgt_hidden_layer[:num_continuing, :, :], Uh[:num_continuing,:,:])
            #import pdb; pdb.set_trace()
            if padding_knockout is not None:
                e_batch += padding_knockout[:num_continuing, :, :]
            alpha_batch = self.softmax(e_batch)
            
            c_batch = torch.sum(
                alpha_batch * src_hidden_seq[:num_continuing, :, :],
                dim=1
            )  #multiplication here is pointwise and broadcast. #result should have dimensions num_continuing by 2*src_hidden_dim
            c_batch=c_batch.view(num_continuing,1,2*self.src_hidden_dim)
            

            #import pdb; pdb.set_trace()
            hidden_out, cur_tgt_hidden_layer = self.decoder(
                    cur_tgt, cur_tgt_hidden_layer[:num_continuing,:,:].transpose(0,1 ), extra_input=c_batch)
            cur_tgt_hidden_layer= cur_tgt_hidden_layer.transpose(0,1)
            # Need to mess around with tranposes to use Afunc.  I keep batchfirst since afunc needs to use a view to combine the n_layers and tgt_hidden_dim dimensions
           # import pdb;  pdb.set_trace()
            hidden_out, _ = rnn.pad_packed_sequence(hidden_out, batch_first=True)  
            #hidden_out should have dimensions num_continuing by 1 by tgt_hidden_dim
            return hidden_out, cur_tgt_hidden_layer


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
        cur_tgt_hidden_layer = Variable(src_hidden_seq.data.new(batchsize, self.decoder.n_layers, self.decoder.hidden_dim).zero_() )

        padding_knockout = Variable(src_hidden_seq.data.new(batchsize, src_max_seq_len,
                                              1).zero_())
        for k in range(batchsize):
            if batch.src.lengths[k]<src_max_seq_len:
                padding_knockout[k, batch.src.lengths[k]:, 0] = -float("inf")

        num_continuing = batchsize  # at the current timestep, the number of sequences that have yet to terminate
        decoder_output = Variable(
            src_hidden_seq.data.new(batchsize, tgt_max_seq_len, self.tgt_hidden_dim).zero_())
        for i in range(tgt_max_seq_len):
            while batch.tgt.lengths[num_continuing - 1] <= i:
                num_continuing -= 1
            assert (num_continuing > 0)
            cur_tgt =batch.tgt.first_k_at_t(k=num_continuing, t=i)
            hidden_out, cur_tgt_hidden_layer  =  self.advance(num_continuing=num_continuing, cur_tgt_hidden_layer= cur_tgt_hidden_layer, src_hidden_seq=src_hidden_seq, cur_tgt=cur_tgt,Uh=Uh, padding_knockout=padding_knockout  )
                        
            decoder_output[:num_continuing, i, :] = hidden_out.view(num_continuing,self.tgt_hidden_dim)
        out = self.lin_out(decoder_output)  
        # out should have dimension batch_size by tgt_max_seq_len by tgt_vocab_size
        goal = torch.cat(
            (batch.tgt.seqs[:, 1:],
             Variable(batch.tgt.seqs.data.new(batchsize, 1).zero_())),
            1)  #prediction is staggered.  at sequence element t we predict t+1
        return self.loss(out.view(-1,self.tgt_vocab_size), goal.view(-1))

    def predict(self, in_seq):
        cuda = next(self.parameters()).is_cuda
        in_seq_len = [len(in_seq)]
        in_seq = dp.TranslationBatch(
            Variable(torch.LongTensor(in_seq)).view(1, -1), in_seq_len)
        if cuda:
            in_seq = in_seq.cuda()
        src_hidden_seq, _ = self.encoder(in_seq)
        src_hidden_seq,_ = rnn.pad_packed_sequence(src_hidden_seq, batch_first=True)
        Uh = self.U(
            src_hidden_seq
        )
        cur_tgt_hidden_layer = Variable(src_hidden_seq.data.new(1, self.decoder.n_layers, self.decoder.hidden_dim).zero_())
        num_continuing=1
        
        index = Variable(torch.LongTensor([lang.SOS_TOKEN]).view(1, 1))
        if cuda:
            index = index.cuda()
       
        indexes=[index.data[0][0]]
        iter=0
        while True:
           cur_tgt=dp.TranslationBatch(index,[1])
           hidden_out, cur_tgt_hidden_layer  =  self.advance(num_continuing=1, cur_tgt_hidden_layer= cur_tgt_hidden_layer, src_hidden_seq=src_hidden_seq,cur_tgt=cur_tgt  ,Uh=Uh, padding_knockout=None )
           weights=self.lin_out(hidden_out).squeeze()#dimension vocab_size 
           #import pdb; pdb.set_trace()
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
class AFunc(nn.Module):
    '''
        Input:
            --tgt_hidden_past: Variable with dimenisons num_continuing by n_layers by tgt_hidden_dim. Provides the value for batches in range(num_continuing)  of the last hidden layer of the decoder in the previous timestep (i.e. timestep i-1 in the paper).
            --Uh: Variable with dimensions num_continuing by src_max_sequence_length by n_layers*tgt_hidden_dim
        Returns:
            -- Variable with dimension num_continuing by src_max_seq_len by 1.  Its (k,j) entry is the value of e_{ij} for the kth batch.  See pg 3 in  Bahdanau et al. 
    '''

    def __init__(self, tgt_hidden_dim,n_layers):
        super(AFunc, self).__init__()
        self.tgt_hidden_dim=tgt_hidden_dim
        self.n_layers=n_layers
        self.v = nn.Linear(n_layers*tgt_hidden_dim, 1, bias=False) #since we are softmaxing after getting the output of v, bias will have no effect
        self.W = nn.Linear(n_layers*tgt_hidden_dim, n_layers*tgt_hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, tgt_hidden_past, Uh):
        #import pdb; pdb.set_trace()
        if self.n_layers>1: #if we are using more than one hidden layer in the decoder, need to concatinate their hidden states to feed to the attention function
            tgt_hidden_past=tgt_hidden_past.contiguous()
            tgt_hidden_past=tgt_hidden_past.view(-1,1,self.n_layers*self.tgt_hidden_dim)
        return self.v(self.tanh(self.W(tgt_hidden_past) + Uh))
