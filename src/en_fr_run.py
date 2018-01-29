import logging
import lang
import manage
man=manage.Manager.basic_enc_dec_from_file("../data/eng-fra_tut/eng-fra.txt",report_interval=1000,l1_name="eng",l2_name="fr",loglevel=logging.DEBUG,batchsize=32,testphrase="They are great.",cuda=True,hidden_dim=512, filt=lang.filter_pair_tut, opt='rmsprop')
man.trainer.train(70000)

