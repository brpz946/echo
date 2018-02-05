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
        logging.info("Constructed a BleuValidator with a validation set of size"+ str(len(self.to_translate)) )   

    def score(self, predictor):
       '''
       Returns:
        -The negative of the bleu score of the predicted translations with respect to the reference translations.
       '''
       predicted=[]
       logging.debug("Starting Prediction")
       for phrase in self.to_translate:
           predicted.append(predictor.predict(phrase))
       logging.debug("prediction complete")
       predicted=util.remove_sos_eos2(predicted)
       logging.info("Calculating BLEU score on validation set.")
       if self.missing_grams(predicted):
           logging.info("Missing n-grams for some n.  Assigining Bleu score of 0")
           return 0
       neg_bleuscore= -bleu.corpus_bleu(self.reference, predicted)
       logging.info("Done calculating BLEU score.")
       return neg_bleuscore 

    def __str__(self):
       return "(negative) BleuValidator"
   

    def missing_grams(self, predicted):
       '''
       Returns true if there are no matching n-grams for some n in {1,2,3,4}
       '''
       for i in range(1,5):
           total_match=0
           for ref, pred in zip(self.reference,predicted):
               frac=bleu.modified_precision(ref,pred,i)
               total_match+=frac.numerator
           if total_match==0:
                return True
       return False
