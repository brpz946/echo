import unittest
import torch
import torch.nn.utils.rnn as rnn
import torch.autograd as ag
import torch.optim as optim
import logging

import util
import data_proc as dp
import trainers as tr
import lang
import enc_dec as ed
import manage


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
        self.assertEqual(batches[0].src.seqs.data.tolist(), [[1, 2, 3], [62, 0, 0]])
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
        encoder = ed.EncoderRNN(5, 6, 7)
        input_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        batch = dp.TranslationBatch(input_padded, [3, 1])
        code = encoder(batch)
        #print(code)
        decoder = ed.DecoderRNN(5, 6, 7)
        correct_output_padded = ag.Variable(
            torch.LongTensor([[1, 2, 3, 4], [1, 0, 0, 0]]))
        batch2 = dp.TranslationBatch(correct_output_padded, [4, 1])
        out = decoder(batch2, code)

    def test_basic_encoder_decoder2(self):
        '''The encoder_decoder should produce output of the right shae '''
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
            in_embedding_dim=100,
            out_embedding_dim=100,
            hidden_dim=100, n_layers=2)
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
            in_embedding_dim=100,
            out_embedding_dim=100,
            hidden_dim=100, n_layers=1, bidirectional= True)
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
            in_embedding_dim=100,
            out_embedding_dim=100,
            hidden_dim=100)

    def test_train_step(self):
        '''
       training the encoder-decoder for one step should change its parameters.
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

    def test_step_gpu(self):
        '''
       training the encoder-decoder for one step should change its parameters.
       '''
        if not torch.cuda.is_available():
            print('skipping GPU test for lack of cuda')
            return
        self.model = self.model.cuda()
        trainer = tr.Trainer(
            self.model, self.ds, 32, 1, reporter=None, cuda=True)
        before = []
        beforeshapes = []
        for param in self.model.parameters():
            before.append(param.data.tolist())
            beforeshapes.append(param.shape)
        trainer.train(1)
        for i, param in enumerate(self.model.parameters()):
            self.assertEqual(param.shape, beforeshapes[i])
            self.assertNotEqual(param.data.tolist(), before[i])


class FileIoTests(unittest.TestCase):
    def test_process_lines(self):
        l1, l2, spairs = lang.read_langsv1('eng', 'fra',
                                           '../data/eng-fra_tut/eng-fra.txt')
        print('example lines:')
        print(spairs[0][0])
        print(spairs[0][1])
        lang.index_words_from_pairs(l1, l2, spairs)
        ipairs = lang.spairs_to_ipairs(spairs, l1, l2)
        pbs, perms, lengths = lang.ipairs_to_padded_batch_seqs(ipairs, 32)


class ManagerTests(unittest.TestCase):
    def test_basic_run(self):
        '''The model should learn to translate when the dataset consists of one phrase '''
        man = manage.Manager.basic_enc_dec_from_file(
            "../data/by_the_gods.txt", loglevel=logging.WARNING)
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
            "../data/by_the_gods.txt", loglevel=logging.WARNING,cuda=True)
        man.trainer.train(100)
        dexsamp = man.l1.sentence2dex("by the gods !")
        pred = man.model.predict(dexsamp)
        translation = man.l2.dex2sentence(pred)
        self.assertEquals(translation, "by the gods !")



if __name__ == '__main__':
    lang_test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        LangTest)
    lang_util_test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        LangUtilTest)
    enc = unittest.defaultTestLoader.loadTestsFromTestCase(EncDecTest)
    io = unittest.defaultTestLoader.loadTestsFromTestCase(FileIoTests)
    pred = unittest.defaultTestLoader.loadTestsFromTestCase(
        EncDecPredictionTests)
    dptest = unittest.defaultTestLoader.loadTestsFromTestCase(DataProcTest)
    trtest = unittest.defaultTestLoader.loadTestsFromTestCase(Trainer_Tests)
    mantests = unittest.defaultTestLoader.loadTestsFromTestCase(ManagerTests)
    multitests= unittest.defaultTestLoader.loadTestsFromTestCase(MultiLayerTrainerTest) 
    bitests= unittest.defaultTestLoader.loadTestsFromTestCase(BiTrainerTests) 
    fast = unittest.TestSuite()
    fast.addTest(lang_test_suite)
    fast.addTest(lang_util_test_suite)
    unittest.TextTestRunner().run(fast)
    #unittest.TextTestRunner().run(io)
    unittest.TextTestRunner().run(enc)
    unittest.TextTestRunner().run(dptest)
    #unittest.TextTestRunner().run(trtest)
    unittest.TextTestRunner().run(pred)
    unittest.TextTestRunner().run(mantests)
    unittest.TextTestRunner().run(bitests)
    #unittest.TextTestRunner().run(multitests)
