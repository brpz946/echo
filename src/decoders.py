import torch
import torch.nn as nn
class OneStepAttnRNN(nn.Module):
    def __init__(self,score_func):
        self.score_func=score_func 

    def forward(self,prev_hidden, in_val, src_hidden_seq, first):
        

