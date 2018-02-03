import logging
import torch.optim as optim
import time


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
    '''

    def __init__(self,
                 model,
                 lr,
                 dataset,
                 batchsize,
                 report_interval=100,
                 cuda=False,
                 reporter=None,
                 opt='sgd'):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.batchsize = batchsize
        self.report_interval = report_interval
        self.reporter = reporter
        self.cuda = cuda
        self.opt = opt

    def train(self, steps):
        if self.opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.opt == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters())
            logging.debug("using rmsprop optimizer")
        batches = self.dataset.batch(self.batchsize)
        i = 0
        num_batches = len(batches)
        starttime = time.time()
        interval_loss = 0
        logging.info("Startng Training")
        for step in range(steps):
            #   print("current_batch=",batches[i])
            loss = train_step(
                batch=batches[i],
                model=self.model,
                optimizer=optimizer,
                cuda=self.cuda)
            interval_loss += loss
            i += 1
            if ((step + 1) %
                    self.report_interval == 0) and self.reporter is not None:
                self.reporter.report(starttime, step + 1, steps,
                                     interval_loss / self.report_interval)
                interval_loss = 0

            if i >= num_batches:
                logging.debug('Reached end of data.  Reshuffling')
                i = 0
                batches = self.dataset.batch(self.batchsize, shuffle=True)

        return self.model
