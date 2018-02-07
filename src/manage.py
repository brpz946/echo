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
class Manager():
    '''
        Conceptually, this class is responsible for for organizing a run 
    '''

    def __init__(self, l1, l2, model, trainer):
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.trainer = trainer

    def save(self, path):
        torch.save(l1, path + "_l1")
        torch.save(l2, path + "_l2")
        torch.save(model, path + "_model")
        torch.save(trainer, path + "_trainer")

    @staticmethod
    def load(path):
        l1 = torch.load(path + "_l1")
        l2 = torch.load(path + "_l2")
        model = torch.load(path + "_model")
        trainer = torch.load(path + "_trainer")
        return Manager(l1, l2, model, trainer)

    @staticmethod
    def basic_from_file(path,
                        lr=0.01,
                        cuda=False,
                        report_interval=10,
                        batchsize=32,
                        in_dim=100,
                        out_dim=100,
                        hidden_dim=100,
                        l1_name="l1",
                        l2_name="l2",
                        testphrase="By the gods!",
                        loglevel=logging.DEBUG,
                        filt=None,
                        opt='sgd',
                        pretrained=False,
                        pre_src_path=None,
                        pre_tgt_path=None,
                        model_constructor=ed.EncoderDecoderRNN.construct,
                        validate=False,
                        record_path=None,
                        model_path=None,
                        dropout=0.2):
        logging.getLogger().setLevel(loglevel)
        l1, l2, spairs = lang.read_langsv1(l1_name, l2_name, path, filt)
        lang.index_words_from_pairs(l1, l2, spairs)
        dataset = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        #todo: add pretrained support here
                
        model = model_constructor(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=in_dim,
            tgt_embedding_dim=out_dim,
            hidden_dim=hidden_dim,dropout=dropout)
        if cuda:
            model = model.cuda()
         
        if validate: 
            validation_data=dataset.split(0.1)
            pred=model.beam_predictor()
            validators=[validation.BleuValidator(validation_data)] 
        else:
            pred=None
            validators=None


        trainer = tr.Trainer(
            model=model,
            lr=lr,
            dataset=dataset,
            batchsize=batchsize,
            report_interval=report_interval,
            cuda=cuda,
            reporter=reporting.TestPhraseReporter(model, l1, l2, testphrase),
            opt=opt,
            predictor=pred,
            validators=validators,record_path=record_path, model_path=model_path)
        return Manager(l1, l2, model, trainer)

    @staticmethod
    def basic_enc_dec_from_file(*largs, **kwargs):
        kwargs["model_constructor"] = ed.EncoderDecoderRNN.construct
        return Manager.basic_from_file(*largs, **kwargs)

    @staticmethod
    def basic_search_from_file(**args):
        args["model_constructor"] = search_rnn.SearchRNN.construct
        return Manager.basic_from_file(**args)
