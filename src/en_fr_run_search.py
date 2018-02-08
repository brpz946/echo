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


mconfig =manage.ManagerConfig() 
mconfig.path="../data/eng-fra_tut/eng-fra.txt"
mconfig.report_interval=1000
mconfig.l1_name="eng"
mconfig.l2_name="fr"
mconfig.loglevel=logging.DEBUG
mconfig.batchsize=32
mconfig.testphrase="They are great."
mconfig.cuda=cuda
mconfig.hidden_dim=128
mconfig.n_layers=2
mconfig.filt=lang.filter_pair_tut
mconfig.opt='rmsprop'
mconfig.validate=True
mconfig.record_path=record_path
mconfig.model_path=model_path
mconfig.dropout=0.2
man=manage.Manager.basic_from_file(mconfig)
man.trainer.train(70000)
man.save(full_model_path)
