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
            for raw_batch in data_loader:
                code_batch, code_seq_lens, ast_batch, ast_seq_lens, _, _ = raw_batch
                #print(code_batch.shape)
                #print(ast_batch.shape)
                code_outputs, _ = self.model.code_encoder(code_batch, code_seq_lens)
                ast_outputs, _ = self.model.ast_encoder(ast_batch, ast_seq_lens)
                #vecs = vecs.max(1).values if pooling == 'max' else vecs.mean(1)
                #print(code_outputs.shape)
                results_1.append(code_outputs)
                results_2.append(ast_outputs)
            results_1 = torch.cat(results_1,1)
            results_2 = torch.cat(results_2,1)
            results_1 = torch.mean(results_1,1)
            results_2 = torch.mean(results_2,1)
        return results_1,results_2

if __name__ == '__main__':
    model = models.Model(code_vocab_size=44808,
                        ast_vocab_size=61,
                        nl_vocab_size=30000,
                        model_file_path='../pretrain_model/pretrain.pt')
    doc2vec = RNNDoc2Vec(model)
    projects=['dagger','dubbo','ExoPlayer','flink','guava','kafka','spring-boot','spring-framework','spring-security']
    dataset_dir = "../dataset_v2/original/"
    vocab_file_path=(config.code_vocab_path, config.ast_vocab_path, config.nl_vocab_path)
    code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
    code_vocab = utils.load_vocab_pk(code_vocab_path)
    ast_vocab = utils.load_vocab_pk(ast_vocab_path)
    nl_vocab = utils.load_vocab_pk(nl_vocab_path)

    for project in projects:
        print(project)
        dataset=data.CodePtrDataset(code_path=os.path.join(dataset_dir,f'{project}/all_truncated_final.code'),
                                                    ast_path=os.path.join(dataset_dir,f'{project}/all_truncated.sbt'),
                                                    nl_path=os.path.join(dataset_dir,f'{project}/all_truncated_final.comment'))
        dataloader=DataLoader(dataset=dataset, batch_size=32, shuffle=False,
                                            collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                                code_vocab=code_vocab,
                                                                                                ast_vocab=ast_vocab,
                                                                                                nl_vocab=nl_vocab,
                                                                                                size1=360,size2=1005))
        code_output,ast_output = doc2vec.get_vecs(dataloader)
        torch.save(code_output, os.path.join(dataset_dir,f'{project}/all_code_semantic.pt'))
        torch.save(ast_output, os.path.join(dataset_dir,f'{project}/all_ast_semantic.pt'))