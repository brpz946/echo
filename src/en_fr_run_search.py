import torch
import logging
import lang
import manage
cuda=torch.cuda.is_available()
man = manage.Manager.basic_search_from_file(
    path="../data/eng-fra_tut/eng-fra.txt",
    report_interval=1000,
    l1_name="eng",
    l2_name="fr",
    loglevel=logging.DEBUG,
    batchsize=32,
    testphrase="They are great.",
    cuda=cuda,
    hidden_dim=512,
    filt=lang.filter_pair_tut,
    opt='rmsprop',validate=True)
man.trainer.train(70000)
