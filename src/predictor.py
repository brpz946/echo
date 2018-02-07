import util
import torch
from torch.autograd import Variable
import copy

import lang


class BeamPredictor:
    '''
    A class for performing beam search using a sequence to sequence model.
        Args:
            --process_src: A function with:
                Args:
                    --src_sequence: A list holding the value of the input sequence, possibly followed by zero padding.
                    --src_length: The length of the un-padded src sequence.  
                Returns:
                    --src_state:  a variable holding model-dependent information that will be fed to advance_output.
            --advance_tgt: A function with 
                Args:
                   --src_state: a FloatTensor variable. Contents are model-dependent
                   --first:  A boolean indicating whether this is the first iteration.
                   --cur_state: A list of float FloatTensor variables. each element holds state values for each sequence currently in the beam. Should either be empty or  have dimensions w by r, where w is less than or equal to beam width and r is model-dependent.
                   --index: A LongTensor variable with dimension w, where w is less than or equal to beam width.  Stores the indexes currently being added for each sequence in the beam.
                Returns:
                    --A w by v LongTensor variable, where v is the tgt vocabulary size.  Entry (i,k) holds the incremental log probability (negative loss) predicted by the model if index k is the next to be added to the sequence represented by row i. 
                    --a list of  w by r FloatTensor.  Row i is the next state predicted by the model for sequence i.
            --rlist: list.  ith element is  length of the rows in ith element of cur_state.  Model-dependent. 
            --k: The number of predictions to output.
            --w: The beam width
            --max_seq_len: The maximum sequence length that will be explored during the beam search.
            --tgt_vocab_size: size of the target vocabulary
            -cuda: Whether to use cuda.
    '''

    def __init__(self,
                 process_src,
                 advance_tgt,
                 rlist, 
                 tgt_vocab_size,
                 k=1,
                 w=1,
                 max_seq_len=30, 
                 cuda=False):
        self.process_src = process_src
        self.advance_tgt = advance_tgt
        self.rlist = rlist
        self.k=k
        self.w=w
        self.tgt_vocab_size = tgt_vocab_size
        self.max_tgt_seq_len = max_seq_len
        self.cuda = cuda

    def predict(self, src_seq):
        return self.beam_search(src_seq)[0][-1] #We want the sequence scoring highest

    def beam_search(self, src_seq ):
        '''
        Carry out a beam search.
        Args:
            --src_seq:  a list containing a sequence of input values
        Returns:
            -A list of k lists, the predictions produced by the model for the input src_vals
        '''
        k=self.k
        w=self.w
        src_state = self.process_src(src_seq)  #src_state is used only by advance_output.  Thus, its contents need only be acceptable to that function.
        cur_state = [Variable(torch.Tensor(0).fill_(0)) for i in self.rlist]
        incoming_index = Variable(torch.LongTensor([lang.SOS_TOKEN]))
        logprobs = Variable(torch.Tensor(1, 1).fill_(0))
        history = [[lang.SOS_TOKEN]]
        if self.cuda:
            incoming_index = incoming_index.cuda()
            logprobs = logprobs.cuda()
            for state in cur_state:
                state= state.cuda()


        best_terminated = util.FixedHeap(k)
        cur_depth = 1

        while True:
            [step_logprobs, states] = self.advance_tgt(src_state=src_state,first=(cur_depth == 1), cur_state=cur_state,index=incoming_index)
            overall_logprobs = step_logprobs + logprobs  #broadcast
            [top_logprobs, top_inds] = torch.topk(overall_logprobs.view(-1), k=k + w, sorted=True)
            rows = top_inds.div(self.tgt_vocab_size)
            cols = top_inds.remainder(self.tgt_vocab_size)

            cur_depth += 1
            if cur_depth >= self.max_tgt_seq_len:
                z = 0
                while best_terminated.cur_size < k:
                    best_terminated.try_add(history[rows.data[z]] +[cols.data[z]] + [lang.EOS_TOKEN],top_logprobs.data[z])
                    z += 1
                break

            p = 0  #current index in old sequences:
            q = 0  #current index in new  sequences being created for next timestep
            s = 0  #number of terminated seqiences found so far
            new_cur_state =[]
            for i in range(len(self.rlist)):
                new_cur_state.append(Variable(cur_state[i].data.new(w, self.rlist[i]).fill_(0)))
            
            
            new_incoming_index = Variable(incoming_index.data.new(w).fill_(0))
            new_logprobs = Variable(logprobs.data.new(w, 1).fill_(-float("inf")))
            new_history = []
            for i in range(w):
                new_history.append([])

            while True:
                if cols.data[p] == lang.EOS_TOKEN:
                    best_terminated.try_add(
                        history[rows.data[p]] + [cols.data[p]],
                        top_logprobs.data[p])
                    s += 1
                else:
                    for i in range(len(self.rlist)):
                       # print("p:",p,"q:",q,"i:",i, "rows:",rows)
                        statei=states[i]
                        state=statei[rows[p], :].squeeze()
                        new_cur_state[i][q, :] =state
                    
                    new_incoming_index[q] = cols[p]
                    new_logprobs[q, 0] = top_logprobs[p]
                    new_history[q] = history[rows.data[p]] + [cols.data[p]]  #cannot use append here since it works in place
                    q += 1
                p += 1

                if q >= w:
                    break
                if s >= k:  #The k terminated sequences found this timestep are better than any un-terminated sequences we can yet find. So we are done.
                    break
                assert (p < k + w)

            cur_state = new_cur_state
            incoming_index = new_incoming_index
            logprobs = new_logprobs
            #  if incoming_index.data[0]==7236:
            history = new_history
            if best_terminated.cur_size >= k and (q == 0 or best_terminated.min_score() >= logprobs.data[0][0] ): 
                # heap full and worst terminated better than best un-terminated or found no terminated this iteration
                break

        seqs, finallogprobs = best_terminated.to_lists()
        return seqs, finallogprobs


class BatchPredictor:
    '''
    A class for batched greedy search using a sequence to sequence model.
    
        Args:
            --process_src: A function with:
                Args:
                    --src_sequences: A LongTensor.  The ith row is the ith input sequence.  Padded
                    --src_length: A list containing the lengths of the un-padded src sequences. 
                Returns:
                    --src_state: A Variable. The ith row of the list holds model-dependent information about the ith input sequence. Will be fed to advance_tgt.
            --advance_tgt: A function with: 
                Args:
                   --src_states: A Variable, see above.
                   --first:  A boolean indicating whether this is the first iteration.
                   --cur_state: A FloatTensor variable. Holds state values for each so-far unterminated sequence. Should either be empty or  have dimensions f by r, f is the number of sequences still continuing, and r is model-dependent.
                   --index: A LongTensor variable with dimension f.  Stores the indexes currently being added for each continuing sequence
                Returns:
                    --A f by v LongTensor variable, where v is the tgt vocabulary size.  Entry (i,k) holds the incremental log probability (negative loss) predicted by the model that index k is the next to be added to the sequence represented by row i. 
                    --A f by r FloatTensor.  Row i is the next state predicted by the model for sequence i.
            --r: Length  of rows of cur_state.  Model-dependent. 
            --tgt_vocab_size: Size of the target vocabulary
            -cuda: Whether to use cuda.
    '''

    def __init__(self,
                 process_src,
                 advance_tgt,
                 r, 
                 tgt_vocab_size,
                 k=1,
                 w=1,
                 cuda=False):
        self.process_src = process_src
        self.advance_tgt = advance_tgt
        self.r = r
        self.tgt_vocab_size = tgt_vocab_size
        self.cuda = cuda

    def batch_predict(self, src_seqs,src_lengths):
        return self.search(src_seqs,src_lengths) 

    def search(self, src_seqs, src_lengths,max_tgt_seq_len=30,sos_mode=None ):
        '''
        Carry out a batched search.
        Args:
            --src_seqs:  A LongTensor.  The ith row  contain the ith sequence of input values. Padded.
            --src_lengths:A LongTensor.  The lengths of the un-padded sequences in src_seq.
            --max_tgt_seq_len: Integer indicating maximum sequence length.  Output sequences will be padded to this length.
            -sos_mode: for future use
        Returns:
            -A LongTensor Variable. 2D Tensor. Dimension (batchsize, max_tgt_seq_len ).  The ith row is the ith output sequence padded with zeros. 
            -A LongTensor Variable. 3D Tensor. Dimension (batchsize, max_tgt_seq_len, tgt_vocab_size).  Entry [i][j][k] is the log probablities produced by the model for the ith output sequence in the jth step. Padded  
            -Long Tensor of giving the lengths of the un-padded output sequences
            -A Long Tensor giving the final culmulative log probabilities
        '''

        batchsize=len(src_seqs)
       # sos_var=Variable(torch.LongTensor([lang.SOS_TOKEN])).view(1,1) 
        #eos_var=Variable(torch.LongTensor([lang.EOS_TOKEN])).view(1,1)

        src_states = self.process_src(src_seqs,src_lengths)  #src_state is a 2d tensor Variable, such that the ith row is associated with the ith batch.  The actual contents of these components are model-dependent. 

        cur_states = Variable(torch.Tensor(batchsize,self.r).fill_(0))
        logprob_history=Variable(torch.Tensor(batchsize,max_tgt_seq_len,self.tgt_vocab_size  ).fill_(0)  ) #note that logprob_history[0,:,:]=0 since  there is no prediction that leads to the SOS token
        seqs=Variable(torch.LongTensor(batchsize,max_tgt_seq_len).fill_(0) ) #the sequences (padded).  Note that the incoming index is already included
        seqs[:,0]=lang.SOS_TOKEN
        lengths=Variable(torch.LongTensor(batchsize).fill_(1))
        cumulative_logprobs = Variable(torch.Tensor(batchsize, 1).fill_(0)) 
        incoming_index = Variable(torch.LongTensor(batchsize).fill_(lang.SOS_TOKEN) )
        continuing=[i for i in range(batchsize) ] #which sequences are continuing
        cur_length = 1

        if self.cuda:
            cur_states = cur_states.cuda()
            incoming_index = incoming_index.cuda()

            lengths=cur_lengths.cuda()
            seqs=seqs.cuda()
            cumulative_logprobs=cumulative_logprobs.cuda()
            logprob_history = logprob_history.cuda()


        while True:
             #_r denotes a variable whose rows are in the reduces space of continuing sequences, not the original space of all sequences
             #import pdb; pdb.set_trace()
             [step_logprobs_r, states_r] = self.advance_tgt( src_states=src_states[continuing,:], first=(cur_length == 1), cur_states=cur_states[continuing,:],index=incoming_index[continuing] )
             cur_states[continuing,:]=states_r
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


