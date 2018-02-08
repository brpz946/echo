import util
import torch
from torch.autograd import Variable
import copy

import lang


class BatchPredictor:
    '''
    A class for batched greedy search using a sequence to sequence model.
    
        Args:
            --process_src: A function with:
                Args:
                    --src_sequences: A LongTensor.  The ith row is the ith input sequence.  Padded
                    --src_length: A list containing the lengths of the un-padded src sequences. 
                Returns:
                    --src_state: A list of variables. The ith row of each variable the list holds model-dependent information about the ith input sequence. Will be fed to advance_tgt.
            --advance_tgt: A function with: 
                Args:
                   --src_states: A Variable, see above.
                   --first:  A boolean indicating whether this is the first iteration.
                   --cur_state: A list of FloatTensor variables. Holds state values for each so-far unterminated sequence. ith element should have have dimensions f by rlist[i], f is the number of sequences still continuing, and r is model-dependent.
                   --index: A LongTensor variable with dimension f.  Stores the indexes currently being added for each continuing sequence
                Returns:
                    --A f by v LongTensor variable, where v is the tgt vocabulary size.  Entry (i,k) holds the incremental log probability (negative loss) predicted by the model that index k is the next to be added to the sequence represented by row i. 
                    --A list of floattensors.  the ith is f by r[i] FloatTensor.  Row j is the next state predicted by the model for sequence j.
            --rlist: list of integers giving the number of columns in each element of cur_state.  Model-dependent. 
            --max_tgt_seq_len: Integer indicating maximum sequence length.  Output sequences will be padded to this length.
            --tgt_vocab_size: Size of the target vocabulary
            -cuda: Whether to use cuda.
    '''

    def __init__(self,
                 process_src,
                 advance_tgt,
                 rlist, 
                 tgt_vocab_size,
                 max_seq_len=30,
                 cuda=False):
        self.process_src = process_src
        self.advance_tgt = advance_tgt
        self.rlist = rlist
        self.max_tgt_seq_len=max_seq_len
        self.tgt_vocab_size = tgt_vocab_size
        self.cuda = cuda

    def batch_predict(self, src_seqs,src_lengths):
        return self.search(src_seqs,src_lengths) 

    def search(self, src_seqs, src_lengths,sos_mode=None ):
        '''
        Carry out a batched search.
        Args:
            --src_seqs:  (remember to sort by length) A LongTensor.  The ith row  contain the ith sequence of input values. Padded.         
            --src_lengths:A list.  The lengths of the un-padded sequences in src_seq.
            -sos_mode: for future use
        Returns:
            -A LongTensor Variable. 2D Tensor. Dimension (batchsize, max_tgt_seq_len ).  The ith row is the ith output sequence padded with zeros. 
            -A LongTensor Variable. 3D Tensor. Dimension (batchsize, max_tgt_seq_len, tgt_vocab_size).  Entry [i][j][k] is the log probablities produced by the model for the ith output sequence in the jth step. Padded  
            -Long Tensor of giving the lengths of the un-padded output sequences
            -A Long Tensor giving the final culmulative log probabilities
        '''
        max_tgt_seq_len=self.max_tgt_seq_len

        batchsize=src_seqs.shape[0]

        src_states = self.process_src(src_seqs,src_lengths)  #src_state is a 2d tensor Variable, such that the ith row is associated with the ith batch.  The actual contents of these components are model-dependent. 

        cur_states =[]
        for i in range(len(self.rlist)):
            cur_states.append(Variable( torch.Tensor(batchsize,self.rlist[i]).fill_(0)))



        logprob_history=Variable(torch.Tensor(batchsize,max_tgt_seq_len,self.tgt_vocab_size  ).fill_(0)  ) #note that logprob_history[0,:,:]=0 since  there is no prediction that leads to the SOS token
        seqs=Variable(torch.LongTensor(batchsize,max_tgt_seq_len).fill_(0) ) #the sequences (padded).  Note that the incoming index is already included
        seqs[:,0]=lang.SOS_TOKEN
        lengths=Variable(torch.LongTensor(batchsize).fill_(1))
        cumulative_logprobs = Variable(torch.Tensor(batchsize, 1).fill_(0)) 
        incoming_index = Variable(torch.LongTensor(batchsize).fill_(lang.SOS_TOKEN) )
        continuing=[i for i in range(batchsize) ] #which sequences are continuing
        cur_length = 1

        if self.cuda:
            cur_states =    [state.cuda() for state in cur_states]
            incoming_index = incoming_index.cuda()

            lengths=lengths.cuda()
            seqs=seqs.cuda()
            cumulative_logprobs=cumulative_logprobs.cuda()
            logprob_history = logprob_history.cuda()



        while True:
             continuing_src_states= [ state[continuing,:].contiguous().view(len(continuing),-1) for state in src_states ]    
             continuing_cur_states= [ state[continuing,:].contiguous().view(len(continuing),-1) for state in cur_states ]

             [step_logprobs_r, states_r] = self.advance_tgt( src_state=continuing_src_states, first=(cur_length == 1), cur_state=continuing_cur_states,index=incoming_index[continuing] )
            # import pdb; pdb.set_trace()
             for i in range(len(cur_states)):
                cur_states[i][continuing,:]=states_r[i]

             overall_logprobs_r = step_logprobs_r + cumulative_logprobs[continuing]  #broadcast Result has dimension len(continuing) by v
           
             new_continuing=[]
             for i in range(len(continuing)):
                logprob_history[continuing[i],cur_length,:]=step_logprobs_r[i,:]#add the logprob history. Note that the first time-step has no logprob history
                
                if cur_length+1 >= max_tgt_seq_len: #if we reach max depth, terminate current sequence then quit 
                    seqs[continuing[i],cur_length]=lang.EOS_TOKEN
                    lengths[continuing[i]]=cur_length+1
                    cumulative_logprobs[continuing[i]]=overall_logprobs_r[i,lang.EOS_TOKEN]
                    continue


                [top_logprob, top_ind] = torch.topk(overall_logprobs_r[i,:].view(-1), k= 1, sorted=True)
                incoming_index[continuing[i]] = top_ind
                seqs[continuing[i],cur_length]=top_ind
                cumulative_logprobs[continuing[i]]=overall_logprobs_r[i,top_ind.data[0]]
                if top_ind.data[0] == lang.EOS_TOKEN:
                    lengths[continuing[i]]=cur_length+1
                else:
                    new_continuing.append(continuing[i])
             cur_length+=1
             if new_continuing:
                continuing=new_continuing      
             else:
                break

        return seqs, logprob_history, lengths,cumulative_logprobs.squeeze() 


