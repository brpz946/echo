import logging

import reporting
import data_proc as dp
import trainers as tr
import lang
import enc_dec as ed
import search_rnn
import word_vectors


class Manager():
    '''
        Conceptually, this class is responsible for for organizing a run 
    '''

    def __init__(self, l1, l2, model, trainer):
        self.l1 = l1
        self.l2 = l2
        self.model = model
        self.trainer = trainer

    @staticmethod
    def basic_enc_dec_from_file(path,
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
                                loglevel=logging.INFO,
                                filt=None,
                                opt='sgd',
                                pretrained=False,
                                pre_src_path=None,
                                pre_tgt_path=None):
        logging.getLogger().setLevel(loglevel)
        l1, l2, spairs = lang.read_langsv1(l1_name, l2_name, path, filt)
        lang.index_words_from_pairs(l1, l2, spairs)
        dataset = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        if pretrained:
            wv_src = word_vectors.WordVectors.from_file(
                pre_src_path, word_set=set(l1.word2index.keys()))
            wv_tgt = word_vectors.WordVectors.from_file(
                pre_tgt_path, word_set=set(l2.word2index.keys()))
            src_missing = set(l1.word2index.keys()).difference(
                set(wv_src.word2vec.keys()))
            tgt_missing = set(l2.word2index.keys()).difference(
                set(wv_tgt.word2vec.keys()))
        enc_dec = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            in_embedding_dim=in_dim,
            out_embedding_dim=out_dim,
            hidden_dim=hidden_dim)
        if cuda:
            enc_dec = enc_dec.cuda()
        trainer = tr.Trainer(
            model=enc_dec,
            lr=lr,
            dataset=dataset,
            batchsize=batchsize,
            report_interval=report_interval,
            cuda=cuda,
            reporter=reporting.TestPhraseReporter(enc_dec, l1, l2, testphrase),
            opt=opt)
        return Manager(l1, l2, enc_dec, trainer)


    
    @staticmethod
    def basic_search_from_file(path,
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
                                loglevel=logging.INFO,
                                filt=None,
                                opt='sgd',
                                pretrained=False,
                                pre_src_path=None,
                                pre_tgt_path=None):
        logging.getLogger().setLevel(loglevel)
        l1, l2, spairs = lang.read_langsv1(l1_name, l2_name, path, filt)
        lang.index_words_from_pairs(l1, l2, spairs)
        dataset = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        if pretrained:
            wv_src = word_vectors.WordVectors.from_file(
                pre_src_path, word_set=set(l1.word2index.keys()))
            wv_tgt = word_vectors.WordVectors.from_file(
                pre_tgt_path, word_set=set(l2.word2index.keys()))
            src_missing = set(l1.word2index.keys()).difference(
                set(wv_src.word2vec.keys()))
            tgt_missing = set(l2.word2index.keys()).difference(
                set(wv_tgt.word2vec.keys()))
        search = search_rnn.SearchRNN(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=in_dim,
            tgt_embedding_dim=out_dim,
            src_hidden_dim=hidden_dim,
            tgt_hidden_dim=hidden_dim)
        if cuda:
            enc_dec = enc_dec.cuda()
        trainer = tr.Trainer(
            model=search,
            lr=lr,
            dataset=dataset,
            batchsize=batchsize,
            report_interval=report_interval,
            cuda=cuda,
            reporter=reporting.TestPhraseReporter(search, l1, l2, testphrase),
            opt=opt)
        return Manager(l1, l2, search, trainer)
