import torch
import logging
from time import localtime, strftime
import lang
import manage
cuda=torch.cuda.is_available()
ltime=localtime()
record_path="../run_logs/run_log_"+strftime("%Y-%m-%dt-%H-%M-%S",ltime)
model_path= "../run_logs/best_model_"+strftime("%Y-%m-%dt-%H-%M-%S",ltime)
full_model_path="../run_logs/full_model"+strftime("%Y-%m-%dt-%H-%M-%S",ltime)


man = manage.Manager.basic_search_from_file(
    path="../data/eng-fra_tut/eng-fra.txt",
    report_interval=1000,
    l1_name="eng",
    l2_name="fr",
    loglevel=logging.DEBUG,
    batchsize=32,
    testphrase="They are great.",
    cuda=cuda,
    hidden_dim=128,
    n_layers=2,
    filt=lang.filter_pair_tut,
    opt='rmsprop',validate=True,
    record_path=record_path,
    model_path=model_path,dropout=0.2)
man.trainer.train(70000)
man.save(full_model_path)
