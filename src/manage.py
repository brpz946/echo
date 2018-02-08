import logging
import torch

import reporting
import data_proc as dp
import trainers as tr
import lang
import enc_dec as ed
import search_rnn
import word_vectors
import predictor
import validation

class ManagerConfig():
     def __init__(self):
         self.path='../data/eng-fra_tut/eng-fra.txt'
         self.lr=0.01
         self.cuda=False
         self.report_interval=10
         self.batchsize=32
         self.in_dim=100
         self.out_dim=100
         self.hidden_dim=100
         self.report_interval=10
         self.batchsize=32
         self.in_dim=100
         self.out_dim=100
         self.hidden_dim=100
         self.l1_name="l1"
         self.l2_name="l2"
         self.testphrase="By the gods!"
         self.loglevel=logging.WARNING
         self.filt=None
         self.opt='sgd'
         self.pretrained=False
         self.pre_src_path=None
         self.pre_tgt_path=None
         self.model_type="search"
         self.validate=False
         self.record_path=None
         self.model_path=None
         self.dropout=0.2
         self.n_layers=1


class Manager():
    '''
        Conceptually, this class is responsible for for organizing a run 
    '''


    def __init__(self, l1, l2, model, trainer,mconfig):
        '''
            Generally, Managers should be created through static functions, which will in turn call __init__
        '''
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.trainer = trainer
        self.mconfig=mconfig

    def save(self, path):
        torch.save(self.l1, path + "_l1")
        torch.save(self.l2, path + "_l2")
        torch.save(self.model.state_dict(), path + "_model")
        torch.save(self.trainer, path + "_trainer")
        torch.save(self.mconfig,path + "_mconfig")
    def translate(self,string):
        return self.l2.dex2sentence(self.model.predict( self.l1.sentence2dex(string)))

    @staticmethod
    def load(path,trainerload=False):
        l1 = torch.load(path + "_l1")
        l2 = torch.load(path + "_l2")
        mconfig=torch.load(path + "_mconfig")
        manager=Manager.basic_from_file(mconfig)
        manager.model.load_state_dict(torch.load(path + "_model"))
        return manager 



    @staticmethod
    def basic_from_file(mconfig):
         
              
        logging.getLogger().setLevel(mconfig.loglevel)
        l1, l2, spairs = lang.read_langsv1(mconfig.l1_name, mconfig.l2_name, mconfig.path, mconfig.filt)
        lang.index_words_from_pairs(l1, l2, spairs)
        dataset = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        if mconfig.model_type== "enc_dec":
            model_constructor=ed.EncoderDecoderRNN.construct
        elif mconfig.model_type == "search":
            model_constructor=search_rnn.SearchRNN.construct
        else:
            raise Exception("Invalid model type")


        model = model_constructor(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=mconfig.in_dim,
            tgt_embedding_dim=mconfig.out_dim,
            hidden_dim=mconfig.hidden_dim,
            dropout=mconfig.dropout,
            n_layers=mconfig.n_layers)
        if mconfig.cuda:
            model = model.cuda()
         
        if mconfig.validate: 
            validation_data=dataset.split(0.1)
            pred=model.beam_predictor()
            validators=[validation.BleuValidator(validation_data)] 
        else:
            pred=None
            validators=None


        trainer = tr.Trainer(model=model,lr=mconfig.lr,dataset=dataset,
            batchsize=mconfig.batchsize,
            report_interval=mconfig.report_interval,
            cuda=mconfig.cuda,
            reporter=reporting.TestPhraseReporter(model, l1, l2, mconfig.testphrase),
            opt=mconfig.opt,
             predictor=pred,
             validators=validators,record_path=mconfig.record_path, 
             model_path=mconfig.model_path)

        return Manager(l1, l2, model, trainer,mconfig)

    @staticmethod
    def basic_enc_dec_from_file(mconfig):
        mconfig.model_type="enc_dec"        
        return Manager.basic_from_file(mconfig)

    @staticmethod
    def basic_search_from_file(mconfig):
        mconfig.model_type="search"
        return Manager.basic_from_file(mconfig)
