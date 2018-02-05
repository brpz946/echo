import logging
import torch.optim as optim
import torch
import time
from time import localtime, strftime

import reporting
def train_step(batch, model, optimizer, cuda=False):
    optimizer.zero_grad()
    if cuda:
        batch = batch.cuda()

# import pdb; pdb.set_trace()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss.data[0]


class Trainer:
    '''
        Responsible for training standard models
        Args:
            --validators: list of validators, the first of which will be used for early stopping
    '''

    def __init__(self,
                 model,
                 lr,
                 dataset,
                 batchsize,
                 report_interval=100,
                 cuda=False,
                 reporter=None,
                 opt='sgd',
                 predictor=None,
                 validators=None,
                 record_path=None,
                 model_path=None):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.batchsize = batchsize
        self.report_interval = report_interval
        self.reporter = reporter
        self.cuda = cuda
        self.opt = opt
        ltime=localtime()
        if record_path== None:
            record_path="../run_logs/run_log_"+strftime("%Y-%m-%dt-%H-%M-%S",ltime)
        if model_path==None:
            model_path= "../run_logs/best_model_"+strftime("%Y-%m-%dt-%H-%M-%S",ltime)
        self.model_path=model_path
        self.record_path=record_path
        self.validators=validators
        self.predictor=predictor

    def train(self, steps):
        if self.opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            logging.debug("using sgd optimizer.")
        elif self.opt == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters())
            logging.debug("using rmsprop optimizer")
        batches = self.dataset.batch(self.batchsize)
        bestscore=float("inf")
        i = 0
        num_batches = len(batches)
        starttime = time.time()
        interval_loss = 0

        with open(self.record_path,"a") as f:
            f.write(self.record_headings())

        scores,bestscore=self.run_validators(bestscore)
        logging.info("Starting Training")
        if self.cuda:
            logging.info("Cuda is enabled.")
        else:
            logging.info("Cuda is disabled.")
                

        for step in range(steps):
            loss = train_step(batch=batches[i], model=self.model, optimizer=optimizer, cuda=self.cuda)
            interval_loss += loss
            i += 1
            if ((step + 1) % self.report_interval == 0) and self.reporter is not None: 
                avg_loss=interval_loss/ self.report_interval
                self.reporter.report(starttime, step + 1, steps, avg_loss)

                scores,bestscore=self.run_validators(bestscore)
                with open(self.record_path,"a") as f:
                    f.write(self.record_row(step,avg_loss,scores,starttime, steps))
                interval_loss = 0

            if i >= num_batches:
                logging.debug('Reached end of data.  Reshuffling')
                i = 0
                batches = self.dataset.batch(self.batchsize, shuffle=True)
        
        scores,bestscore=self.run_validators(bestscore)
       # import pdb; pdb.set_trace()
        if self.validators is not None:
            self.model.load_state_dict(torch.load(self.model_path))
        return self.model


    def run_validators(self,bestscore):
        scores=[]
        if self.validators is not None:
            for i,validator in enumerate(self.validators):
                scores.append(validator.score(self.predictor))
                    
            if scores[0] < bestscore:
                logging.debug("Current score "+ str(scores[0])+" better than previous best score of "+str(bestscore)+"." )
                logging.info("****Saving Model*****")
                bestscore=scores[0]
                torch.save(self.model.state_dict(),self.model_path) 
            else:
                logging.debug("Current score "+ str(scores[0])+" worse  than previous best score of "+str(bestscore)+"." )

        return (scores,bestscore)

    def record_headings(self):
        string="iterations, percent complete,elapsed time, time remaining, loss"
        if self.validators is not None:
            for validator in self.validators:
                string+=","+str(validator)
        string+="\n"
        return string

    def record_row(self, iteration,loss,scores,starttime,totaliter):
        progress, elapsed, est = reporting.progress_info(starttime, iteration, totaliter)
        string= ",".join([str(iteration), str(progress*100), str(elapsed), str(est), str(loss) ])
        if scores is not None:
            for score in scores:
                string+=","+str(score )
        string+="\n"
        return string
