import heapq
import torch
import torch.nn as nn
import pickle

import lang

def perm_compose(p, q):
    return torch.LongTensor([p[q[i]] for i in range(len(p))])


def perm_invert(p):
    pinv = torch.zeros(len(p)).long()
    for i in range(len(p)):
        pinv[p[i]] = i
    return pinv

class FixedHeap:
    '''
    A heap with a limited capacity.
    '''
    def __init__(self, k):
        self.k = k
        self.cur_size = 0
        self.seen_so_far = 0
        self.heap = []

    def try_add(self, item, score):
        if self.cur_size < self.k:
            heapq.heappush(self.heap, (score, self.seen_so_far, item))
            self.cur_size += 1
        else:
            if score > self.heap[0][0]:
                heapq.heapreplace(self.heap, (score, self.seen_so_far, item))

        assert (len(self.heap) <= self.k)
        self.seen_so_far += 1

    def min_score(self):
        return self.heap[0][0]

    def to_lists(self):
        '''
        Converts the heap to two lists, returing items in increasing order
        '''
        items = []
        scores = []
        while len(self.heap) > 0:
            entry = heapq.heappop(self.heap)
            items.append(entry[2])
            scores.append(entry[0])
        return items, scores

def remove_right_zeros(lst):
    newlst=[]
    index=0
    while True:
        if lst[index]==0  or index== len(lst):
            break
        newlst.append(list[i])
        index+=1
    return newlst

def remove_sos_eos2(sentences):
    ''' 
        Args:
            --A 2 deep list of integers. The element [i][k] is the kth word in the ith phrase. 
        Returns: Same list with leading lang.SOS_TOKEN and trailing lang.EOS_TOKEN removed
    '''
    pruned_sentences=[]
    for sentence in sentences:
        assert(sentence[0]==lang.SOS_TOKEN)
        assert(sentence[len(sentence)-1]==lang.EOS_TOKEN)
        pruned_sentences.append(sentence[1:len(sentence)-1])
    return pruned_sentences 

def remove_sos_eos3(sentences):
    '''
        Args:
            --A 3 deep list of integers. The element [i][j][k] is the kth word in  phrase [i][j]. 
        Returns: Same list with leading lang.SOS_TOKEN and trailing lang.EOS_TOKEN removed
    '''
    out=[]
    for row in sentences:
        out.append( remove_sos_eos2(row) )
    return out

def save_obj(path, name ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path ):
    with open(path, 'rb') as f:
        return pickle.load(f)


