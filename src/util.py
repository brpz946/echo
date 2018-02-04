import heapq
import torch


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
        items = []
        scores = []
        while len(self.heap) > 0:
            entry = heapq.heappop(self.heap)
            items.append(entry[2])
            scores.append(entry[0])
        return items, scores
