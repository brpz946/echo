import logging
import unicodedata
import string
import re
import random
import copy
import torch
import torch.autograd as ag
SOS_TOKEN = 1
EOS_TOKEN = 2
NUM_RESERVED_INDEXES = 3  # null +SOS_TOKEN + EOS_TOKEN


class Lang:
    '''
    A language
    '''

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "NULL", 1: "SOS", 2: "EOS"}
        self.n_words = 3

    def add_seq(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def dex2sentence(self, dexes):
        '''
         Converts a list of indicies into the corresponding string
         Removes SOS and EOS tokens
         '''
        assert (dexes[0] == SOS_TOKEN)
        assert (dexes[len(dexes) - 1] == EOS_TOKEN)
        str = ""
        for count, index in enumerate(dexes):
            if count == 0 or count == len(dexes) - 1:
                continue
            str += self.index2word[index]
            str += " "
        return str.strip()

    def sentence2dex(self, sentence):
        dexes = [
            self.word2index[w] for w in normalize_string(sentence).split(' ')
        ]
        dexes.insert(0, SOS_TOKEN)
        dexes.append(EOS_TOKEN)
        return dexes


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )  #NFD=  decomposition of accented characters into char + accent. Mn= accents


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)  #remove nonstandard chars
    s = re.sub(r"([.!?])", r" \1 ", s)  #seperate punctuation
    s = re.sub(r" +", r" ", s)  #remove redundant spaces
    s = s.strip()
    return s


def read_langsv1(lang1, lang2, path, filt=None, reverse=False):
    '''
        read language data from a file in the format used in the Pytorch tutorial
        http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        
    '''
    logging.info('reading lines... ')
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    logging.info(str(len(lines)) + 'lines read')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    if filt is not None:
        pairs = [pair for pair in pairs if filt(pair)]
        logging.info(str(len(pairs)) + "lines after filtering")
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def index_words_from_pairs(lang1, lang2, pairs):
    for pair in pairs:
        lang1.add_seq(pair[0])
        lang2.add_seq(pair[1])


def spairs_to_ipairs(pairs, lang1, lang2):
    ipairs = [[[lang1.word2index[s1] for s1 in pair[0].split()],
               [lang2.word2index[s2] for s2 in pair[1].split()]]
              for pair in pairs]
    for pair in ipairs:
        for seq in pair:
            seq.insert(0, SOS_TOKEN)
            seq.append(EOS_TOKEN)
    return ipairs


MAX_LENGTH = 10

eng_prefixes = ("i am ", "i m ", "he is", "he s ", "she is", "she s",
                "you are", "you re ", "we are", "we re ", "they are",
                "they re ")


def filter_pair_tut(pair):
    '''
     purpose of this method is to allow replication of the tutorial at
    ://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.htmli   
    '''
    return len(pair[0].split(' ')) < MAX_LENGTH and len(
        pair[1].split(' ')) < MAX_LENGTH and pair[0].startswith(eng_prefixes)
