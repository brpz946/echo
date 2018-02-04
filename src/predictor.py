
import heapq
import torch
from torch.autograd import Variable

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
                   --cur_state: A FloatTensor variable. Holds state values for each sequence currently in the beam. Should either be empty or  have dimensions w by r, where w is less than or equal to beam width and r is model-dependent.
                   --index: A LongTensor variable with dimension w, where w is less than or equal to beam width.  Stores the indexes currently being added for each sequence in the beam.
                Returns:
                    --A w by v LongTensor variable, where v is the tgt vocabulary size.  Entry (i,k) holds the incremental log probability (negative loss) predicted by the model if index k is the next to be added to the sequence represented by row i. 
                    --A w by r FloatTensor.  Row i is the next state predicted by the model for sequence i.
            --r: Length of the rows of rows of cur_state.  Model-dependent. 
            --max_seq_len: The maximum sequence length that will be explored during the beam search.
            --tgt_vocab_size: size of the target vocabulary
    '''
    def __init__(self,process_src, advance_tgt,r,tgt_vocab_size,max_seq_len=30): 
        self.process_src=process_src
        self.advance_tgt=advance_tgt
        self.r=r 
        self.tgt_vocab_size=tgt_vocab_size
        self.max_tgt_seq_len=max_seq_len
        
        
    def predict(self, src_seq, k, w, cuda=False):        
        '''
        Carry out a beam search.
        Args:
            --src_seq:  a list containing a sequence of input values
            --k: The number of predictions to output.
            --w: The beam width
            --cuda: whether to use cuda
        Returns:
            -A list of k lists, the predictions produced by the model for the input src_vals
        '''
        src_state=self.process_src(src_seq) #src_state is used only by advance_output.  Thus, its contents need only be acceptable to that function. 
        cur_state=Variable(torch.Tensor(0).fill_(0))
        incoming_index=Variable(torch.LongTensor([lang.SOS_TOKEN]))
        logprobs=Variable(torch.Tensor(1,1).fill_(0))
        history=[[ lang.SOS_TOKEN  ]]
        if cuda:
            incoming_index=incoming_index.cuda()
            logprobs=logprobs.cuda()
            cur_state=cur_state.cuda()

        best_terminated=FixedHeap(k) 
        cur_depth=1
        
        while True:
            [step_logprobs,states]=self.advance_tgt(src_state=src_state,first=(cur_depth==1),cur_state=cur_state,index=incoming_index)
            overall_logprobs=step_logprobs+ logprobs #broadcast
            [top_logprobs, top_inds]=torch.topk(overall_logprobs.view(-1),k=k+w,sorted=True)
            rows=top_inds.div(self.tgt_vocab_size)
            cols=top_inds.remainder(self.tgt_vocab_size) 
            
            cur_depth+=1
            if cur_depth >= self.max_tgt_seq_len:
                z=0
                while best_terminated.cur_size<k:
                    best_terminated.try_add( history[rows.data[z]] + [cols.data[z]] + [lang.EOS_TOKEN], top_logprobs.data[z]  )
                    z+=1
                break

            p=0 #current index in old sequences
            q=0 #current index in new  sequences being created for next timestep
            s=0 #number of terminated seqiences found so far
            new_cur_state=Variable(cur_state.data.new(w,self.r).fill_(0))
            new_incoming_index=Variable(incoming_index.data.new(w).fill_(0) )
            new_logprobs=Variable(logprobs.data.new(w,1).fill_(-float("inf")))
            new_history=[]
            for i in range(w):
                new_history.append([])

            while True:
                if  cols.data[p] == lang.EOS_TOKEN:
                    best_terminated.try_add( history[rows.data[p]] + [cols.data[p]], top_logprobs.data[p] )
                    s+=1
                else:
                    new_cur_state[q,:]=states[rows[p],:]                           
                    new_incoming_index[q]=cols[p]
                    new_logprobs[q,0]=top_logprobs[p]
                    new_history[q]=history[rows.data[p]]+  [cols.data[p]] #cannot use append here since it works in place
                    q+=1
                p+=1

                if q>=w:
                    break
                if s>=k: #The k terminated sequences found this timestep are better than any un-terminated sequences we can yet find. So we are done.
                    break
                assert(p<k+w)

            cur_state=new_cur_state
            incoming_index=new_incoming_index
            logprobs=new_logprobs
          #  if incoming_index.data[0]==7236:
           #     import pdb; pdb.set_trace()
            history=new_history
            if best_terminated.cur_size >= k and (q==0  or best_terminated.min_score()>=logprobs.data[0][0]): # heap full and worst terminated better than best un-terminated or found no terminated this iteration
                break

        seqs,finallogprobs=best_terminated.to_lists()
        return seqs,finallogprobs


class FixedHeap:
    
    def __init__(self,k):
        self.k=k
        self.cur_size=0
        self.seen_so_far=0
        self.heap=[]

    def try_add(self, item, score):
        if self.cur_size<self.k:
            heapq.heappush(self.heap,(score,self.seen_so_far,item))
            self.cur_size+=1
        else:
            if score> self.heap[0][0]:
                heapq.heapreplace(self.heap,(score,self.seen_so_far,item))

        assert(len(self.heap)<=self.k)
        self.seen_so_far+=1
    
    def min_score(self): 
        return self.heap[0][0]
    
    def to_lists(self):
        items=[]
        scores=[]
        while len(self.heap)>0:
            entry= heapq.heappop(self.heap)
            items.append(entry[2])
            scores.append(entry[0])
        return items,scores
