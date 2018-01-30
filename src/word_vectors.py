import logging
import numpy as np
import torch
import lang

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
        logging.info("Loaded " + str(n_found) + " word_vector from file")
        if local_word_set:
            #did not find all words
            logging.warning("Did not find word vectors for the words: "+ str(local_word_set) )

    def apply_to_weights(self, weights, language, index_set=None):
        if index_set == None:
            index_set=set(range(lang.NUM_RESERVED_INDEXES,language.n_words))
        filtered_index_set={index for index in index_set if language.index2word[index] in self.word2vec}
        for index in filtered_index_set:
            weights[index,:]=self.word2vec[language.index2word[index]]
        return index_set.difference(filtered_index_set)
