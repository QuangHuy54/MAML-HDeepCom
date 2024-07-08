import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler,SGD
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import config
import data
import models
import eval
torch.manual_seed(1)

class RNNDoc2Vec(object):
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
    
    def get_vecs(self, data_loader):
        results_1 = []
        results_2 = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in tqdm(data_loader, total=int(np.ceil(len(data_loader) / self.batch_size)),
                                  desc='Caching model outputs...'):
                code_batch, code_seq_lens, ast_batch, ast_seq_lens, nl_batch, nl_seq_lens = raw_batch
                code_outputs, _ = self.code_encoder(code_batch, code_seq_lens)
                ast_outputs, _ = self.ast_encoder(ast_batch, ast_seq_lens)
                #vecs = vecs.max(1).values if pooling == 'max' else vecs.mean(1)
                results_1.append(code_outputs.to('cpu'))
                results_2.append(ast_outputs.to('cpu'))
            results_1 = torch.cat(results_1)
            results_2 = torch.cat(results_2)
        return results_1,results_2

if __name__ == '__main__':
    # best_model_dict = _train()
    # _test(best_model_dict)
    _test('../pretrain_model/pretrain.pt')