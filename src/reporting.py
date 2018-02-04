import time
import math
import logging

#from http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


def as_minutes(s):
    m = math.floor(s / 60)
    secs = s - m * 60
    return '%dm %ds' % (m, secs)


def time_since(since, progress):
    now = time.time()
    s = now - since
    if progress == 0:
        return asMinutes(s), "unknown"
    es = s / (progress)
    rs = es - s
    return as_minutes(s), as_minutes(rs)

def progress_info(starttime, curiter, totaliter):
    progress= curiter/totaliter
    elapsed,estimated_remaining=time_since(starttime, progress)
    return (progress, elapsed, estimated_remaining)


class Reporter():
    '''
        Reporters are responsible for recording or displaying information during training
    '''

    def report(self, starttime, curiter, totaliter, loss):
        progress, elapsed, est = progress_info(starttime, curiter, totaliter)
        s = "Elapsed time: %s. Iteration: %d. Progress: %d. Remaining: %s. loss: %.6f " % (
            elapsed, curiter, 100 * progress, est, loss)
        logging.info(s)


class TestPhraseReporter(Reporter):
    '''
    
    '''

    def __init__(self, model, l1, l2, phrases):
        self.model = model
        self.l1 = l1
        self.l2 = l2
        if isinstance(phrases, str):
            phrases = [phrases]
        self.phrases = phrases

    def report(self, starttime, curiter, totaliter, loss):
        super().report(starttime, curiter, totaliter, loss)
        for phrase in self.phrases:
            dexsamp = self.l1.sentence2dex(phrase)
            pred = self.model.predict(dexsamp)
            logging.info("Test phrase: %s", phrase)
            #logging.info("Code:%s", dexsamp)
            #logging.info("Mapped code:%s", pred)
            logging.info("Translation:%s", self.l2.dex2sentence(pred))
