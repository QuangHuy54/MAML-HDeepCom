from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from learn2learn.algorithms import MAML
import os
import time

import models
import data
import utils
import config

class Translator(object):
    def __init__(self, model, sampling_temp=sampling_temp, sampling_topk=sampling_topk,
                 beam_width=beam_width, n_best=n_best, max_iter=max_iter,
                 length_penalty=length_penalty, coverage_penalty=coverage_penalty,
                 metrics=['loss', 'bleu'], unk_replace=False, smooth=3):
        self.model = model
        self.metrics = metrics
        self.unk_replace = unk_replace
        self.smooth = smooth
        

    @property
    def metrics(self):
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        metrics = set(metrics)
        all_metrics = {'loss', 'bleu', 'rouge', 'meteor'}
        if not metrics.issubset(all_metrics):
            raise ValueError('Unkown metric(s): ' + str(metrics.difference(all_metrics)))
        self._metrics = metrics
    
    def val_loss(self, final_state, memory, src_lengths, tgt_batch, weights):
        outputs, _ = self.model.decoder(tgt_batch[:,:-1], final_state, memory, src_lengths)
        loss = sequence_loss(outputs, tgt_batch[:,1:], is_probs=self.model.is_ensemble,
                             pad_id=self.wrapped_decoder.pad_id)
        return loss.item()
    
    def translate_batch(self, src_batch, src_lengths=None, tgt_batch=None, *args):
        weights = args[0] if self.weights else None
        raw_batches = args[-1] if self.raw_data else [None]
        reports = dict(scores=None, attn_history=None)
        with torch.no_grad():
            memory, final_state = self.model.encoder(src_batch, src_lengths)
            if 'loss' in self._metrics:
                reports['loss'] = self.val_loss(final_state, memory, src_lengths, tgt_batch, weights)
            predicts, reports['scores'], reports['attn_history'] = \
            self.wrapped_decoder(final_state, memory, src_lengths, weights)
        
        
        predicts = id2word(predicts, self.model.decoder.field,
                           (raw_batches[0], reports['attn_history']), 
                           replace_unk=self.unk_replace)
        if 'bleu' in self._metrics:
            reports['bleu'] = batch_bleu(predicts, raw_batches[-1], self.smooth) * 100
        predicts = [' '.join(s) for s in predicts]
        
        if not self._metrics.isdisjoint({'rouge', 'meteor'}):
            targets = [' '.join(s) for s in raw_batches[-1]]
            if 'rouge' in self._metrics:
                rouge = batch_rouge(predicts, targets)
                reports['rouge'] = rouge['rouge-l']['f'] * 100
            if 'meteor' in self._metrics:
                reports['meteor'] = batch_meteor(predicts, targets) * 100
        return predicts, reports
    
    def eval_one_batch(self, batch, batch_size, criterion):
        """
        evaluate one batch
        :param batch:
        :param batch_size:
        :param criterion:
        :return:
        """
        with torch.no_grad():

            # code_batch and ast_batch: [T, B]
            # nl_batch is raw data, [B, T] in list
            # nl_seq_lens is None
            nl_batch = batch[4]

            decoder_outputs = self.model(batch, batch_size, self.nl_vocab)  # [T, B, nl_vocab_size]

            decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
            nl_batch = nl_batch.view(-1)

            loss = criterion(decoder_outputs, nl_batch)

            return loss
    def init_generator(self, data_gen):
        data_gen.raw_data = self.unk_replace or self._metrics.difference({'loss'})
        self.raw_data = data_gen.raw_data
        self.weights = self.model.is_ensemble and data_gen.weights
    
    def __call__(self, batches, save_path=None):
        self.init_generator(batches)
        self.model.eval()
        results = []
        reports = defaultdict(float, scores=[], attn_history=[])
        
        pbar = tqdm(batches, desc='Translating...')
        for batch in pbar:
            predicts, reports_ = self.translate_batch(*batch)
            pbar.set_postfix({metric: reports_[metric] for metric in self._metrics})
            results.extend(predicts)
            for metric in self._metrics:
                reports[metric] += reports_[metric]
#            reports['scores'].extend(reports_['scores'])
            # reports['attn_history'].extend(reports_['attn_history'])
        
        for metric in self._metrics:
            reports[metric] /= len(batches)
            print('total {}: {:.2f}'.format(metric, reports[metric]))
            if metric == 'loss':
                reports['ppl'] = perplexity(reports[metric])
                print('total ppl: {:.2f}'.format(reports['ppl']))
        if save_path is not None:
            save(results, save_path)
        return results, reports
    
