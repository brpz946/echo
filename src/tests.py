import unittest
import torch
import torch.nn.utils.rnn as rnn
import torch.autograd as ag
from torch.autograd import Variable
import torch.optim as optim
import logging
import math

import util
import data_proc as dp
import trainers as tr
import lang
import enc_dec as ed
import search_rnn
import manage
import word_vectors as wv
import basic_rnn
import predictor as pr
import validation as val
class LangTest(unittest.TestCase):
    def test_add_word(self):
        '''Words added to a language should appear in word2index, word2count, index2word '''
        l = lang.Lang('testl')
        b = l.n_words
        l.add_word('foo')
        l.add_word('bar')
        self.assertEqual(l.word2index['foo'], b)
        self.assertEqual(l.word2index['bar'], b + 1)
        self.assertEqual(l.word2count['foo'], 1)
        self.assertEqual(l.word2count['bar'], 1)
        self.assertEqual(l.index2word[3], 'foo')
        self.assertEqual(l.index2word[4], 'bar')

    def test_add_dup_word(self):
        '''Lang should detect duplicate words'''
        l = lang.Lang('testl')
        b = l.n_words
        l.add_word('the')
        l.add_word('the')
        l.add_word('the')
        l.add_word('the')
        l.add_word('the')
        l.add_word('the')
        self.assertEqual(l.word2count['the'], 6)
        self.assertEqual(l.n_words, b + 1)

    def test_sentence2dex(self):
        l = lang.Lang('testl')
        b = l.n_words
        l.add_word('havoc')
        l.add_word('cry')
        l.add_word('!')
        self.assertEquals(
            l.sentence2dex("cry havoc!"),
            [lang.SOS_TOKEN, b + 1, b, b + 2, lang.EOS_TOKEN])
        self.assertEquals(
            l.dex2sentence(l.sentence2dex("cry havoc!")), "cry havoc !")


class DataProcTest(unittest.TestCase):
    def test_supervised_translation_dataset_from_string(self):
        '''
        factory method should properly construct a SupervisedTranslationDataset
        '''
        l1 = lang.Lang('testl1')
        b = l1.n_words
        l2 = lang.Lang('testl2')
        l1.add_word('king')
        l1.add_word('queen')
        l2.add_word('roi')
        l2.add_word('renne')
        entry = [["king queen", "renne roi"]]
        ds = dp.SupervisedTranslationDataset.from_strings(entry, l1, l2)
        self.assertEqual(ds.lseq,
                         [[[lang.SOS_TOKEN, b, b + 1, lang.EOS_TOKEN],
                           [lang.SOS_TOKEN, b + 1, b, lang.EOS_TOKEN]]])

    def test_batch(self):
        '''should properly convert lists of lists of lists of word indicies to lists of lists of padded batch sequences  '''
        testpairs = [[[1, 2, 3], [1]], [[62], [1, 2]], [[11, 12], [0, 2]]]
        ds1 = dp.SupervisedTranslationDataset(testpairs)
        batches = ds1.batch(2)
        self.assertEqual(batches[0].src.seqs.data.tolist(),
                         [[1, 2, 3], [62, 0, 0]])
        self.assertEqual(batches[0].tgt.seqs.data.tolist(), [[1, 2], [1, 0]])
        self.assertEqual(batches[1].src.seqs.data.tolist(), [[11, 12]])
        self.assertEqual(batches[0].perm.tolist(), [1, 0])
        testpairs2 = [[[1, 2, 3], [4]], [[62], [1, 2]], [[11, 12], [4, 5, 6]]]
        ds2 = dp.SupervisedTranslationDataset(testpairs2)
        batches = ds2.batch(3)
        self.assertEqual(batches[0].src.seqs.data.tolist(),
                         [[1, 2, 3], [11, 12, 0], [62, 0, 0]])
        self.assertEqual(batches[0].tgt.seqs.data.tolist(),
                         [[4, 5, 6], [1, 2, 0], [4, 0, 0]])
        self.assertEqual(batches[0].perm.tolist(), [1, 2, 0])
        self.assertEqual(batches[0].src.lengths, [3, 2, 1])
        self.assertEqual(batches[0].tgt.lengths, [3, 2, 1])


class LangUtilTest(unittest.TestCase):
    def test_unicode_to_ascii(self):
        ''' unicodeToAscii should strip accents '''
        s = 'âêîô'
        t = 'aeio'
        self.assertEqual(lang.unicode_to_ascii(s), t)

    def test_normalize_string(self):
        '''normalizeString should lower case letters and replace special chacters by spaces '''
        s = "A@d#f$G!%"
        t = "a d f g !"
        self.assertEqual(lang.normalize_string(s), t)

    def test_spairs_to_ipairs(self):
        '''spairs_to_ipairs should replace lists of lists of strings with lists of lists of lists of word indicies'''
        l1 = lang.Lang('testl1')
        b = l1.n_words
        l2 = lang.Lang('testl2')
        l1.add_word('king')
        l1.add_word('queen')
        l2.add_word('roi')
        l2.add_word('renne')
        entry = [["king queen", "renne roi"]]
        self.assertEqual(
            lang.spairs_to_ipairs(entry, l1, l2),
            [[[lang.SOS_TOKEN, b, b + 1, lang.EOS_TOKEN],
              [lang.SOS_TOKEN, b + 1, b, lang.EOS_TOKEN]]])

    def test_perm_compose(self):
        p = torch.LongTensor([2, 1, 0])
        q = torch.LongTensor([1, 0, 2])
        r = torch.LongTensor([1, 2, 0])
        self.assertEqual(util.perm_compose(p, q).tolist(), r.tolist())

    def test_perm_invert(self):
        q = torch.LongTensor([1, 0])
        self.assertEqual(util.perm_invert(q).tolist(), q.tolist())


class EncDecTest(unittest.TestCase):
    def test_basic_encoder_decoder(self):
        '''encoder and decoder should process a batch without an error  '''
        encoder = basic_rnn.RNN(5, 6, 7)
        input_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        batch = dp.TranslationBatch(input_padded, [3, 1])
        _, code = encoder(batch.seqs,lengths=batch.lengths)
        #print(code)
        decoder = basic_rnn.RNN(5, 6, 7)
        correct_output_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        batch2 = dp.TranslationBatch(correct_output_padded, [4, 1])
        out, _ = decoder(batch2.seqs, code,lengths=batch2.lengths)

    def test_rnn_with_extra_input(self):
        '''  
        When used with extra input, an rnn should process a batch without error
        '''
        encoder = basic_rnn.RNN(5, 6, 7, extra_input_dim=1)
        input_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        extra_in = ag.Variable(torch.Tensor(2, 4, 1))
        extra_in.requires_grad = True
        batch = dp.TranslationBatch(input_padded, [3, 1])
        _, code = encoder(batch.seqs, extra_input=extra_in,lengths=batch.lengths)
        #

    def test_basic_encoder_decoder2(self):
        '''The encoder_decoder should produce output of the right shape '''
        encdec = ed.EncoderDecoderRNN(5, 5, 6, 6, 7)
        input_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        correct_output_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        tbatch1 = dp.TranslationBatch(input_padded, [4, 1])
        tbatch2 = dp.TranslationBatch(correct_output_padded, [4, 1])
        stb = dp.SupervisedTranslationBatch(tbatch1, tbatch2,
                                            ag.Variable(
                                                torch.LongTensor([0, 1])))
        result = encdec(stb)
        self.assertEqual(result.shape[0], (1))


        #for name, param in encdec.named_parameters():
        #    print('Name=', name)
        #    print(param)
class EncDecPredictionTests(unittest.TestCase):
    def setUp(self):
        L = 1000
        self.encdec = ed.EncoderDecoderRNN(4, 4, 4, 4, 4)
        for name, param in self.encdec.named_parameters():
            param.data = torch.zeros_like(param.data)
        for name, param in self.encdec.encoder.gru.named_parameters():
            if name == "bias_ih_l0":
                param.data = torch.Tensor([0, 0, 0, 0, L, L, L, L, 0, 0, 0, 0])
        for name, param in self.encdec.decoder.gru.named_parameters():
            if name == "weight_ih_l0":
                param.data = torch.Tensor(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, L], [0, L, 0, 0]])
            if name == "bias_ih_l0":
                param.data = torch.Tensor(
                    [0, 0, 0, 0, -L, -L, -L, -L, 0, 0, 0, 0])
        self.encdec.encoder.embedding.weight.data = torch.eye(4)
        self.encdec.decoder.embedding.weight.data = torch.eye(4)
        self.encdec.lin.weight.data = torch.eye(4)

    def test_enc_dec_pred(self):
        '''
            For fixed values of its weights, the encoder-decoder should predict according to its governing equations 
        '''
        pred = self.encdec.predict([1, 2, 3])
        self.assertEquals([1, 3, 2], pred)

    def test_enc_dec_pred_cuda(self):
        '''
            For fixed values of its weights, the encoder-decoder should predict according to its governing equations 
            Tests with Cuda enabled
        '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return
        self.encdec = self.encdec.cuda()
        #   import pdb; pdb.set_trace()
        pred = self.encdec.predict([1, 2, 3])
        self.assertEquals([1, 3, 2], pred)


class MultiLayerTrainerTest(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            hidden_dim=100,
            n_layers=2)

    def test_train_step(self):
        '''
       training a multi-layer encoder-decoder for one step should change its parameters.
       '''
        trainer = tr.Trainer(self.model, 0.01, self.ds, 32, 1, reporter=None)
        before = []
        beforeshapes = []
        for param in self.model.parameters():
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])


class BiTrainerTests(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            hidden_dim=100,
            n_layers=1,
            bidirectional=True)

    def test_train_step(self):
        '''
       training a bidirectional encoder-decoder for one step should change its parameters.
       '''
        trainer = tr.Trainer(self.model, 0.01, self.ds, 32, 1, reporter=None)
        before = []
        beforeshapes = []
        for param in self.model.parameters():
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])


class Trainer_Tests(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            hidden_dim=100)

    def test_train_step(self):
        '''
       training the encoder-decoder for one step should change its parameters.
       '''
        trainer = tr.Trainer(
            model=self.model,
            lr=0.01,
            dataset=self.ds,
            batchsize=32,
            reporter=None)
        before = []
        beforeshapes = []
        for param in self.model.parameters():
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])

    def test_step_gpu(self):
        '''
       training the encoder-decoder for one step should change its parameters.
       '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return
        self.model = self.model.cuda()
        trainer = tr.Trainer(
            model=self.model,
            lr=0.01,
            dataset=self.ds,
            batchsize=32,
            reporter=None,
            cuda=True)
        before = []
        beforeshapes = []
        for param in self.model.parameters():
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])


class ManagerTestsPretrained(unittest.TestCase):
    def test_basic_run_pretrained_gpu(self):
        '''The model should learn to translate when the dataset consists of one phrase when it uses pretrained word vectors and runs on the gpu '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return
        man = manage.Manager.basic_enc_dec_from_file(
            "../data/testing/by_the_gods_dustballs.txt",
            loglevel=logging.WARNING,
            pretrained=True,
            pre_src_path='../data/fastText_word_vectors/wiki.en.vec',
            pre_tgt_path='../data/fastText_word_vectors/wiki.en.vec',
            cuda=True)
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods ! dustballs !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods ! dustballs !")

    def test_basic_run_pretrained(self):
        '''The model should learn to translate when the dataset consists of one phrase when it uses pretrained word vectors  '''
        man = manage.Manager.basic_enc_dec_from_file(
            "../data/testing/by_the_gods_dustballs.txt",
            loglevel=logging.WARNING,
            pretrained=True,
            pre_src_path='../data/fastText_word_vectors/wiki.en.vec',
            pre_tgt_path='../data/fastText_word_vectors/wiki.en.vec')
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods ! dustballs !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods ! dustballs !")


class ManagerTests(unittest.TestCase):
    def test_basic_run(self):
        '''The model should learn to translate when the dataset consists of one phrase '''
        man = manage.Manager.basic_enc_dec_from_file(
            "../data/testing/by_the_gods.txt", loglevel=logging.WARNING,validate=False)
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods !")

    def test_basic_run_gpu(self):
        '''The model should learn to translate when the dataset consists of one phrase when trained on the gpu '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return

        man = manage.Manager.basic_enc_dec_from_file(
            "../data/testing/by_the_gods.txt",
            loglevel=logging.WARNING,
            cuda=True, validate=False)
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods !")


class WordVectorTests(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        self.l1, self.l2, self.spairs = lang.read_langsv1(
            'eng', 'fra', '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(self.l1, self.l2, self.spairs)
        self.wvec = wv.WordVectors.from_file(
            '../data/fastText_word_vectors/wiki.en.vec',
            word_set=set(self.l1.word2index.keys()))

    def test_embedding_load(self):
        '''
            Simple test of loading word vectors from a file and organizing them they can be used in a translation model
        '''
        missing_words = set(self.l1.word2index.keys()).difference(
            self.wvec.word2vec.keys())
        logging.debug("missing words are: " + str(missing_words))
        pt = self.wvec.produce_embedding_vecs(self.l1, missing_words)
        sosvec = torch.zeros(302)
        sosvec[300] = 1
        self.assertEqual(pt.missing_dict[lang.SOS_TOKEN], 0)
        self.assertEqual(pt.missing_dict[lang.EOS_TOKEN], 1)
        self.assertEqual(sosvec[300:302].tolist(),
                         pt.missing[0, 300:302].data.tolist())
        self.assertFalse(pt.weights.requires_grad)
        self.assertTrue(pt.missing.requires_grad)


class SearchRNNFastTests(unittest.TestCase):
    def test_simp(self):
        sch = search_rnn.SearchRNN(5, 5, 2, 3, 6, 7)
        src_batch = dp.TranslationBatch(
            Variable(torch.LongTensor([[4, 3, 2], [1, 0, 0]])), [3, 1])
        tgt_batch = dp.TranslationBatch(
            Variable(torch.LongTensor([[3, 4, 0], [3, 3, 3]])), [2, 3])
        batch = dp.SupervisedTranslationBatch(src_batch, tgt_batch,
                                              torch.LongTensor([0, 1]))
        sch(batch)


class SearchRNNSlowTests(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = search_rnn.SearchRNN(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            src_hidden_dim=100,
            tgt_hidden_dim=100,
            n_layers=2)

    def test_search_train_step(self):
        '''
       training a multi-layer encoder-decoder for one step should change its parameters.
       '''
        trainer = tr.Trainer(self.model, 0.01, self.ds, 32, 1, reporter=None)
        before = []
        names = []
        beforeshapes = []
        for name, param in self.model.named_parameters():
         #   print("parameter: ", name)
            names.append(name)
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
          #  print("comapring parameter ", names[i])
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])


class MoreSearchRNNTests(unittest.TestCase):
    # def test_search_basic_run(self):
    # '''The model should learn to translate when the dataset consists of one phrase u '''

    # man = manage.Manager.basic_search_from_file(
    # "../data/testing/by_the_gods.txt", loglevel=logging.WARNING )
    # man.trainer.train(100)
    # dexsamp = man.l1.sentence2dex("by the gods !")
    # pred = man.model.predict(dexsamp)
    # translation = man.l2.dex2sentence(pred)
    # self.assertEquals(translation, "by the gods !")

    def test_search_basic_run(self):
        '''The model should learn to translate when the dataset consists of one phrase  on the gpu '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return


        man = manage.Manager.basic_search_from_file(
            path="../data/testing/by_the_gods.txt",
            loglevel=logging.WARNING,
            cuda=True)
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods !")


#    def test_search_basic_run2(self):
#        '''The model should learn to translate when the dataset consists of a few simple phrases '''
#
#        man = manage.Manager.basic_search_from_file(
#            "../data/testing/alpha_beta.txt", loglevel=logging.INFO,testphrase=["a","b", "d", "a b", "b d", "d a", "d b a"], opt= 'rmsprop', batchsize=3 )
#        man.trainer.train(10000)
#        dexsamp = man.l1.sentence2dex("d b a")
#        pred = man.model.predict(dexsamp)
#        translation = man.l2.dex2sentence(pred)
#        self.assertEquals(translation, "delta beta alpha")


class PredictorTests(unittest.TestCase):
    '''
        Beam Search should behave as expected when given synthetic data.
    '''

    def test_beam_search(self):
        def process_src(src):
            return []

        def advance_tgt(src_state, first, cur_state, index):
            num_seqs = index.shape[0]
            probs = Variable(torch.Tensor(num_seqs, 6).fill_(0))
            for i in range(num_seqs):
                if index.data[i] == 1:
                    probs[i, :] = torch.log(
                        Variable(torch.Tensor([0, 0, 0, 0.6, 0.4, 0])))
                elif index.data[i] == 3:
                    probs[i, :] = torch.log(
                        Variable(torch.Tensor([0, 0, 0.9, 0, 0, 0.1])))
                elif index.data[i] == 4:
                    probs[i, :] = torch.log(
                        Variable(torch.Tensor([0, 0, 1, 0, 0, 0])))
                elif index.data[i] == 5:
                    probs[i, :] = torch.log(
                        Variable(torch.Tensor([0, 0, 0, 0, 0, 1])))
                else:
                    self.assertEqual(0, 1)
            stateout = Variable(torch.Tensor(num_seqs, 1))
            return probs, stateout

        predictor = pr.BeamPredictor(
            process_src, advance_tgt, r=1, max_seq_len=20, tgt_vocab_size=6,k=2,w=2)
        seqs, probs = predictor.beam_search(src_seq=[])
        self.assertEqual(seqs[0], [1, 4, 2])
        self.assertEqual(seqs[1], [1, 3, 2])
        self.assertAlmostEquals(probs[0], math.log(0.4))
        self.assertAlmostEqual(probs[1], math.log(0.54))


class MorePredictorTests(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            hidden_dim=100)
        self.model2 = search_rnn.SearchRNN(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            src_hidden_dim=100,
            tgt_hidden_dim=100,
            n_layers=2)

    # def test_enc_dec_beam_consistancy(self):
    # '''
    # BeamPredictor with beam width 1 should produce the same result as the old greedy prediction function
    # '''
    # beam=self.model.beam_predictor()
    # oldpred= self.model.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN])
    # newpred=beam.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN],k=1,w=1 )[0]
    # # import pdb; pdb.set_trace()
    # self.assertEqual(oldpred,newpred[0] )

    # def test_search_beam_consistancy(self):
    # '''
    # BeamPredictor with beam width 1 should produce the same result as the old greedy prediction function when applied to SearchRNN
    # '''
    # beam=self.model2.beam_predictor()
    # oldpred= self.model2.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN])
    # newpred=beam.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN],k=1,w=1 )[0]
    # #  import pdb; pdb.set_trace()
    # self.assertEqual(oldpred,newpred[0] )

    # def test_search_beam_consistancy_gpu(self):
    # '''
    # BeamPredictor with beam width 1 should produce the same result as the old greedy prediction function when applied to SearchRNN on the GPU
    # '''
    # self.model2=self.model2.cuda()
    # beam=self.model2.beam_predictor()
    # oldpred= self.model2.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN])
    # newpred=beam.predict([lang.SOS_TOKEN,5,lang.EOS_TOKEN],k=1,w=1 )[0]
    # #  import pdb; pdb.set_trace()
    # self.assertEqual(oldpred,newpred[0] )
class DummyPredictor:
    def predict(self,seq):
        if seq == [1,3,2]:
            return [1,6,7,8,9,10,2]
        elif seq == [1,4,2]:
            return [1,55,55,55,55,55,2]

class DummyPredictor2:
     def predict(self,seq):
        if seq == [1,3,2]:
            return [1,6,7,8,9,10,2]
        elif seq == [1,4,2]:
            return [1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,2]


class DummyPredictor3:
     def predict(self,seq):
        if seq == [1,3,2]:
            return [1,6,7,2]
        elif seq == [1,4,2]:
            return [1,7,8,2]



class BleuValidation(unittest.TestCase):
    def test_bleu_val(self):
        data=[ [[1,3,2],  [1,6,7,8,9,10,2]],
             [[1,4,2],  [1,7,8,9,10,11,12,13,14,15,16,17,18,19,20,2]] 
                ]
    
        validator=val.BleuValidator(data)
        self.assertNotEqual(validator.score(DummyPredictor()),-1)
        self.assertEqual(validator.score(DummyPredictor2()),-1)
        self.assertEqual(validator.score(DummyPredictor3() ),0)


class BatchPredTests(unittest.TestCase):

    def test_batch_search(self):
        '''
            Batch search should behave as expected when applied to synthetic data
        '''
        def process_src(src_sec,src_len):
            return src_sec

        def advance_tgt(src_states,first, cur_states, index):
            num_seqs=index.shape[0]
            probs = Variable(torch.Tensor(num_seqs, 6).fill_(0))
            for i in range(num_seqs):
                if src_states.data[i,0]==3:
                    if index.data[i] == 1:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0, 0.6, 0.4, 0])))
                    elif index.data[i] == 3:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0.9, 0, 0, 0.1])))
                    elif index.data[i] == 4:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 1, 0, 0, 0])))
                    elif index.data[i] == 5:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0, 0, 0, 1])))
                    else:
                        self.assertEqual(0, 1)
                elif src_states.data[i,0]==4:
                    if index.data[i] == 1:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0, 0.4, 0.6, 0])))
                    elif index.data[i] == 3:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0.9, 0, 0, 0.1])))
                    elif index.data[i] == 4:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 1, 0, 0, 0])))
                    elif index.data[i] == 5:
                        probs[i, :] = torch.log(Variable(torch.Tensor([0, 0, 0, 0, 0, 1])))
                    else:
                        self.assertEqual(0, 1)
                elif src_states.data[i,0] == 5:
                    probs[i,:]= torch.log(Variable(torch.Tensor([0, 0, 1, 0, 0, 0])))
                else :
                    self.assertEqual(0,1)
            stateout = Variable(torch.Tensor(num_seqs, 1))
            return probs, stateout

        predictor = pr.BatchPredictor(process_src,advance_tgt, r=1, tgt_vocab_size=6)
        seqs, logprob_history,lengths,logprobs = predictor.search(src_seqs=Variable(torch.Tensor([[3],[4],[5]])), src_lengths=Variable(torch.Tensor([1,1,1])))

        l=lengths.data.tolist()
        self.assertEqual(l,[3,3,2])
        self.assertEqual(seqs[0][:l[0]].data.tolist(), [1, 3, 2])
        self.assertEqual(seqs[1][:l[1]].data.tolist(), [1, 4, 2])
        self.assertEqual(seqs[2][:l[2]].data.tolist(), [1,  2])
        self.assertAlmostEquals(logprobs.data[0], math.log(0.54))
        self.assertAlmostEqual(logprobs.data[1], math.log(0.6))
        self.assertAlmostEqual(logprobs.data[2], 0)


class BatchPredTests2(unittest.TestCase):
    def setUp(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(l1, l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(spairs, l1, l2)
        self.model = ed.EncoderDecoderRNN(
            l1.n_words,
            l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            hidden_dim=100)
        self.model2 = search_rnn.SearchRNN(
            src_vocab_size=l1.n_words,
            tgt_vocab_size=l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            src_hidden_dim=100,
            tgt_hidden_dim=100,
            n_layers=2)


class NoPackRNNTests(unittest.TestCase):
    def test_basic(self):
        rnn=basic_rnn.RNN(vocab_size=5,embedding_dim=1,hidden_dim=1,pack=False)
        rnn(Variable( torch.LongTensor([1]).view(1,1) ),lengths=[1] )


if __name__ == '__main__':
    lang_test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        LangTest)
    lang_util_test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        LangUtilTest)
    enc = unittest.defaultTestLoader.loadTestsFromTestCase(EncDecTest)
    pred = unittest.defaultTestLoader.loadTestsFromTestCase(
        EncDecPredictionTests)
    dptest = unittest.defaultTestLoader.loadTestsFromTestCase(DataProcTest)
    trtest = unittest.defaultTestLoader.loadTestsFromTestCase(Trainer_Tests)
    mantests = unittest.defaultTestLoader.loadTestsFromTestCase(ManagerTests)
    premantests = unittest.defaultTestLoader.loadTestsFromTestCase(
        ManagerTestsPretrained)
    multitests = unittest.defaultTestLoader.loadTestsFromTestCase(
        MultiLayerTrainerTest)
    bitests = unittest.defaultTestLoader.loadTestsFromTestCase(BiTrainerTests)
    wvtests = unittest.defaultTestLoader.loadTestsFromTestCase(WordVectorTests)
    schtests = unittest.defaultTestLoader.loadTestsFromTestCase(
        SearchRNNFastTests)
    schslowtests = unittest.defaultTestLoader.loadTestsFromTestCase(
        SearchRNNSlowTests)
    schmore = unittest.defaultTestLoader.loadTestsFromTestCase(
        MoreSearchRNNTests)
    beam = unittest.defaultTestLoader.loadTestsFromTestCase(PredictorTests)
    morebeam = unittest.defaultTestLoader.loadTestsFromTestCase(
        MorePredictorTests)
    bleuval = unittest.defaultTestLoader.loadTestsFromTestCase(BleuValidation)
    bp = unittest.defaultTestLoader.loadTestsFromTestCase(BatchPredTests)
    nopack = unittest.defaultTestLoader.loadTestsFromTestCase(NoPackRNNTests)
    fast = unittest.TestSuite()
    fast.addTest(lang_test_suite)
    fast.addTest(lang_util_test_suite)
    unittest.TextTestRunner().run(fast)
    unittest.TextTestRunner().run(enc)
    unittest.TextTestRunner().run(dptest)
    unittest.TextTestRunner().run(pred)
    unittest.TextTestRunner().run(mantests)
    unittest.TextTestRunner().run(trtest)  #slow
    unittest.TextTestRunner().run(premantests)  #slow
    unittest.TextTestRunner().run(bitests)  #slow
    unittest.TextTestRunner().run(multitests)  #slow
    unittest.TextTestRunner().run(wvtests)  #slow
    unittest.TextTestRunner().run(schslowtests) #sloW
    unittest.TextTestRunner().run(schtests)
    unittest.TextTestRunner().run(schmore)
    unittest.TextTestRunner().run(beam)
    logging.getLogger().setLevel(logging.DEBUG)
   # unittest.TextTestRunner().run(bleuval)
    unittest.TextTestRunner().run(bp)
    unittest.TextTestRunner().run(nopack)
    
