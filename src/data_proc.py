import random
import copy
import math
import torch
from torch.autograd import Variable

import lang
import util


class TranslationBatch:
    '''
    A batch of sequences
        Atributes:
            -seqs: A varible continaing a 2d LongTensor.  the [k,l] element is the index correspoding to the lth token in the kth sequence.  Since they may be of different length, the sequences are zero-padded. Sequences should be sorted in descending order by length
            -lengths:  list of integers containing lengths of the sequences in seqs.
    '''

    def __init__(self, seqs, lengths):
        self.seqs = seqs
        self.lengths = lengths

    def cuda(self):
        return TranslationBatch(self.seqs.cuda(), self.lengths)

    @staticmethod
    def from_list(lseqs):
        '''
        Args:
            --lseqs: a 2-deep list of integers
     '''
        batchsize = len(lseqs)
        seq_lens = torch.LongTensor([len(seq) for seq in lseqs])
        pad_tensor = torch.zeros(batchsize, max(seq_lens)).long()
        for k in range(batchsize):
            pad_tensor[k, :seq_lens[k]] = torch.LongTensor(lseqs[k])
        _, perm = seq_lens.sort(0, descending=True)
        pad_tensor = pad_tensor[perm]
        translation_batch = TranslationBatch(
            Variable(pad_tensor), seq_lens[perm].tolist())
        return translation_batch, perm

    def first_k(self, k):
        return TranslationBatch(self.seqs[:k, :], self.lengths[:k])

    def first_k_at_t(self, k, t):
        assert (self.lengths[k - 1] > t)
        newseq = self.seqs[:k, t]
        newseq.contiguous()
        #import pdb; pdb.set_trace()
        return TranslationBatch(
            newseq.view(k, 1),
            torch.LongTensor(k).fill_(1).tolist())

    def __str__(self):
        s = "TranslationBatch containing:\n" + str(self.seqs)
        return s


class SupervisedTranslationBatch:
    '''
    two batches of sequences
    Atributes:
        --src: a TranslationBatch from the source language
        --tgt: A TranslationBatch from the target language
        --perm: the sequences in the batches are sorted by length, not alligned. Applying this permutation to the src sequences aligns them with the tgt sequences
        
    '''

    def __init__(self, src, tgt, perm):
        self.src = src
        self.tgt = tgt
        self.perm = perm

    def cuda(self):
        return SupervisedTranslationBatch(self.src.cuda(), self.tgt.cuda(),
                                          self.perm.cuda())

    @staticmethod
    def from_list(lseqs):
        '''
            Args:
                --lseqs: a 3-deep list of integers whose  [i][j][k] element is the index of the  kth word of the jth sequence in the ith language in the batch. Note that this ordering is different from that of SupervisedTranslationDataset!
        '''
        src, perm_src = TranslationBatch.from_list(lseqs[0])
        tgt, perm_tgt = TranslationBatch.from_list(lseqs[1])
        perm = util.perm_compose(util.perm_invert(perm_src), perm_tgt)
        return SupervisedTranslationBatch(src, tgt, perm)

    def __str__(self):
        #  import pdb; pdb.set_trace()
        s = "SuperVisedTranslationBatch containing source:\n" + str(
            self.src) + "\n and target: \n" + str(
                self.tgt) + "\n and and perm:\n" + str(self.perm)
        return s


class SupervisedTranslationDataset:
    '''
    A dataset of aligned sequences
    Attributes:
        --lseq: a 3-deep list of integers, whose [i][j][k] element is the index of the kth word in ith phrase when expressed in the jth language
        
     '''

    def __init__(self, lseq):
        self.lseq = lseq

    @staticmethod
    def from_strings(pairs, lang1, lang2):
        '''
            Args:
                --pairs: a 2-deep list of strings Whose [i][j] string is the ith phrase in the jth language
                --lang1: The first language
                --lang2: The second language
        '''

        ipairs = [[[lang1.word2index[s1] for s1 in pair[0].split()],
                   [lang2.word2index[s2] for s2 in pair[1].split()]]
                  for pair in pairs]
        for pair in ipairs:
            for seq in pair:
                seq.insert(0, lang.SOS_TOKEN)
                seq.append(lang.EOS_TOKEN)
        return SupervisedTranslationDataset(ipairs)

    def batch(self, batchsize, shuffle=False):
        '''
            Returns:
                --A list of supervised translation batches suitable for feeding into a model
        '''
        ip = copy.deepcopy(self.lseq)
        batches = []
        if shuffle:
            random.shuffle(ip)
        for i in range((len(ip) - 1) // batchsize + 1):
            curbatchsize = min((i + 1) * batchsize, len(ip)) - i * batchsize
            seqs = [[ip[l][j] for l in range(i * batchsize, i * batchsize + curbatchsize) ] for j in [0, 1]] #reorder the sequences so [j][i][k] refers to kth word in ith sentence i jth language
            batches.append(SupervisedTranslationBatch.from_list(seqs))
        return batches

    def split(self, prop):
        '''
        Extracts a speicified proprtion of the data and returns it in head.  After this function can been called on a dataset, it contains all data not returned in head.
        Args:
            --prop: a number between 0 and 1
            --shuffle: a boolean.   Whether to suffle the data before extrating head.
        Returns:
            --Head: A 3-deep list consisting of  proportion prop of the data
        '''
        random.shuffle(self.lseq)
        head_size = math.floor(prop * len(self.lseq))
        assert(head_size>0)
        head = self.lseq[:head_size]
        lseq = self.lseq[head_size:]
        return head
