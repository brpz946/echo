import logging
import numpy as np
import torch
import lang
from torch.nn import Parameter
PRINT_INTERVAL=20000
class WordVectors:
    def __init__(self):
        self.word2vec={}
    def load_from_file(self, path, max_iter=-1,  word_set=None, format="fasttext"):
        logging.info("Loading word vectors from file: " + path)
        if word_set is not None:
            using_word_set=True
            local_word_set=word_set.copy()
        else:
            using_word_set=False
        n_found=0
        if format == "fasttext":
            with open(path) as f:
                # based on https://github.com/facebookresearch/MUSE/blob/master/src/utils.py 
                for i, line in enumerate(f):
                    if i ==0 :
                        self.dimension = int(line.split()[1])
                        continue
                    if max_iter>=0 and i >= max_iter:
                        break
                    if using_word_set and not local_word_set: #the local word set is empty  
                        break
                    if i % PRINT_INTERVAL == 0:
                        logging.debug("Read "+str(i) +" lines so far.  Found "+str(n_found)+ " word vectors.") 

                        if using_word_set==True:
                            logging.debug("Missing "+ str(len(local_word_set)) +" word vectors.")
                    word, vec = line.rstrip().split(' ', 1)

                    if using_word_set: 
                        if word not in local_word_set:
                            continue
                        else:
                            local_word_set.remove(word)
                    vec =  np.fromstring(vec, sep=' ')
                    if np.linalg.norm(vec) ==0:
                        vec[0] = 0.01
                    self.word2vec[word]= torch.Tensor(vec)
                    n_found+=1
        else:
            raise Exception("Unknown Format")
        logging.info("Loaded " + str(n_found) + " word vectors from file")

        if using_word_set:
            if local_word_set:
                logging.warning("Did not find word vectors for the words: "+ str(local_word_set) )
        return local_word_set

    def produce_embedding_vecs(self,language, missing_word_set ):
        '''
        Intended to be used with language models using pretrained word vectors.
        Args:
            -language: the language from which the words in word2vec are drawn
            -missing_word_set: The set of words in the language for which we do not have word vectors
        Returns:
        -weights:   A Parameter whose ith row is the vector corresponding to the word indexed by i in language. Rows corresponding to words in missing_word_set are zero. requires_grad is turned off, as the vectors are pretrained.  Note that we embed the d dimesnional word vectors in a d+2 dimensional space.  These extra dimensions are intended to allow the symbols SOS and EOS to be distinct from standard words
        -missing: a Paramter  whose ith row is a tensor corresponding to the ith missing word.  requires_grad is turned on, since we do not have word vectors for these words.
        -missing_dict: a dictionary associating the indexes of the missing words in language to their row in missing
    '''
        weights=Parameter(torch.cat( (self.apply_to_weights(torch.zeros(language.n_words, self.dimension), language),torch.zeros(language.n_words,lang.NUM_RESERVED_INDEXES-1 ) ),1 ))
        weights.requires_grad=False #pretrained
        
        missing=torch.Tensor(len(missing_word_set)+lang.NUM_RESERVED_INDEXES-1, self.dimension+lang.NUM_RESERVED_INDEXES -1)
        # the final two dimensions start zero expcept for for reserved symbols. Note that we do not give a vector to the null symbol,since it will never show up in sequence we process
        missing[:,-lang.NUM_RESERVED_INDEXES+1:]=0         
        for i in range(lang.NUM_RESERVED_INDEXES-1 ):
            missing[i,self.dimension+i]=1
        missing=Parameter(missing)
        missing.requires_grad=True
        missing_dict={}
        for i in range(lang.NUM_RESERVED_INDEXES-1):
            missing_dict[i+1]=i
        j=lang.NUM_RESERVED_INDEXES
        for word in missing_word_set:
            missing_dict[language.word2index[word]]=j
            j=j+1
        return weights, missing, missing_dict             

    def apply_to_weights(self, weights, language, index_set=None):
        if index_set == None:
            index_set=set(range(lang.NUM_RESERVED_INDEXES,language.n_words))
        filtered_index_set={index for index in index_set if language.index2word[index] in self.word2vec}
        for index in filtered_index_set:
            weights[index,:]=self.word2vec[language.index2word[index]]
        return weights

class pretrained_embedding:
    def __init__(self,weights,missing,missing_dict):
        self.weights=weights
        self.missing=missing
        self.missing_dict=missing_dict
