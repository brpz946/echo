import logging
import nltk.translate.bleu_score as bleu

import util
class BleuValidator:
    '''
    Scores a predictor by its negative bleu score with reference data 
    Attributes:
        --to_translate: 2-deep list containing phrases to be translated.  Includes SOS and EOS tokens
        --reference: 3 deep list of reference phrases.  SOS and EOS tokens removed
    
    Args:
            --validation_data: A 3-deep list of integers. Entry [i][0][k] is the kth word in the ith phrase to be translated.   Entry [i][j][k] for j>0 is the kth word in the jth reference phrase for the ith translation phrase. Asumed to have SOS and EOS tokens.        
    '''

    def __init__(self,validation_data):
        self.to_translate= [ [ word for word in row[0] ] for row in validation_data   ]
        self.reference=  [ [ [  word for word in  col  ] for col in row[1:] ] for row in validation_data] 
        self.reference=util.remove_sos_eos3(self.reference)
           

    def score(self, predictor):
       '''
       Returns:
        -The negative of the bleu score of the predicted translations with respect to the reference translations.
       '''
       predicted=[]
       for phrase in self.to_translate:
           predicted.append(predictor.predict(phrase))
       predicted=util.remove_sos_eos2(predicted)
       logging.info("Calculating BLEU score on validation set.")
       neg_bleuscore= -bleu.corpus_bleu(self.reference, predicted)
       logging.info("Done calculating BLEU score.")
       return neg_bleuscore 

    def __str__(self):
       return "(negative) BleuValidator"
