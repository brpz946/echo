
import heapq

import lang
import torch
from torch.autograd import Variable
#note to self: two competing goals:  batch prediction and beam search. I guess I will go with beam search.  it is ok if we are less efficient when predicting, since validation is infrequent.  So should be ok
#another problem: GPU/CPU transfers.  Generally want to avoid this.  But if I work with lists, I cant. But actually probably ok a few times per iteration, since I am computing softmax

class BeamPredictor:
    '''
        Args:
            --process_src: a function that accepts a src sequence (a list) and produces a src state (A variable tensor), which will be provided to the advance_output function 
            --advance_output: A function with 
                Args:
                   --input_state: a FloatTensor variable. Contents are model-dependent
                   --curstate: A FloatTensor variable. Holds state values for each sequence currently in the beam. Should either be None (iteration 0). Or have dimensions w by r, where w is beam width and r is model-dependent
                   --index: A LongTensor variable with dimension w.  Stores the indicies about to be added for each senence in the beam.
                Returns:
                    --a w by v longtensor variable, where v is the tgt vocabulary size
    '''
    def __init__(self,process_input, advance_output,max_seq_len=30): 
        self.process_input=process_input
        self.advance_output=advance_output
        self.max_seq_len=max_seq_len
        
        
        
    def predict(self, src_vals, k, w,i cuda)        
        '''
        Args:
            --src_vals:  a list containing a sequence of input values
            --k: The number of predictions to output.
            --w: The beam width
            --cuda: whether to use cuda
            Returns:
            -A list of k lists, the predictions produced by the model for the input src_vals
        '''
        src_state=process_src(src_vals) #src_state is a tensor used by advance_output.  Thus, its dimensions and contents are up to that functon
        
        best_terminated=[] 
        explored_count=0
        current_depth=0
        
        probs=Variable()

        while True:
            pass
