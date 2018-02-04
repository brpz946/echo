import torch
import torch.autograd

import lang
import data_proc as dp
import enc_dec as ed
import trainers as tr
import search_rnn


class Profiler:
    def setup(self):
        self.l1, self.l2, spairs = lang.read_langsv1(
            'eng', 'fra', '../data/eng-fra_tut/eng-fra.txt')
        lang.index_words_from_pairs(self.l1, self.l2, spairs)
        self.ds = dp.SupervisedTranslationDataset.from_strings(
            spairs, self.l1, self.l2)

    def profile(self, model, out_path):
        '''
        Outputs profiler results as a text file and as a chrome tracing.  To access the chrome tracing, open chrome and enter chrome://tracing in the URL bar.  Then click load and open the tracing file.
        '''
        trainer = tr.Trainer(model, 0.01, self.ds, 32, 1, reporter=None)
        with torch.autograd.profiler.profile() as prof:
            trainer.train(1)
        prof.export_chrome_trace(out_path + "_tracing")
        tablestring = prof.table(sort_by='cpu_time_total')
        rows = tablestring.splitlines()
        body_rows = rows[3:]
        flipped_body_rows = list(reversed(body_rows))
        out_table = "\n".join(rows[:3]) + "\n" + "\n".join(flipped_body_rows)
        with open(out_path + "_table.txt", "w") as f:
            f.write(out_table)

    def profile_enc_dec_cpu(self, out_path="../profiler_results/enc_dec_cpu"):
        self.setup()
        model = ed.EncoderDecoderRNN(
            self.l1.n_words,
            self.l2.n_words,
            in_embedding_dim=100,
            out_embedding_dim=100,
            hidden_dim=100,
            n_layers=2)
        self.profile(model, out_path)

    def profile_search_cpu(self, out_path="../profiler_results/search_cpu"):
        self.setup()
        model = search_rnn.SearchRNN(
            src_vocab_size=self.l1.n_words,
            tgt_vocab_size=self.l2.n_words,
            src_embedding_dim=100,
            tgt_embedding_dim=100,
            src_hidden_dim=100,
            tgt_hidden_dim=100,
            n_layers=1)
        self.profile(model, out_path)


if __name__ == "__main__":
    profiler = Profiler()
    #profiler.profile_search_cpu()
    profiler.profile_enc_dec_cpu()
