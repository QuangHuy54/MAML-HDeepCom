import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading
import random
import matplotlib.pyplot as plt
import numpy as np
import utils
import config
import data
import models
import eval
from tqdm import tqdm
import learn2learn as l2l
import train

def tuple_map(fn, t, **kwargs):
    if t is None:
        return None
    if type(t) not in {list, tuple}:
        return fn(t, **kwargs)
    return tuple(tuple_map(fn, s, **kwargs) for s in t)

class MetaTrain(object):

    def __init__(self, training_projects, validating_project, vocab_file_path=None, model_file_path=None):
        """

        :param vocab_file_path: tuple of code vocab, ast vocab, nl vocab, if given, build vocab by given path
        :param model_file_path:
        """

        self.training_projects = training_projects
        self.validating_project = validating_project
        self.vocab_file_path=vocab_file_path
        # dataset
        dataset_dir = "../dataset_v2/original/"
        self.meta_datasets = {}
        for project in (training_projects + [validating_project]):
            self.meta_datasets[project]={
                "support": data.CodePtrDataset(code_path=os.path.join(dataset_dir,f'{project}/all_truncated.code'),
                                                ast_path=os.path.join(dataset_dir,f'{project}/all_truncated.sbt'),
                                                nl_path=os.path.join(dataset_dir,f'{project}/all_truncated.comment')),
                "query": data.CodePtrDataset(code_path=os.path.join(dataset_dir,f'{project}/valid.code'),
                                                ast_path=os.path.join(dataset_dir,f'{project}/valid.sbt'),
                                                nl_path=os.path.join(dataset_dir,f'{project}/valid.comment'))
            }
        
        self.meta_datasets_size = sum([(len(dataset['support'])) for dataset in self.meta_datasets.values()])

        self.meta_dataloaders = {}
        for project in training_projects:
            self.meta_dataloaders[project] = {
                'support': DataLoader(dataset=self.meta_datasets[project]['support'], batch_size=config.support_batch_size, shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab,
                                                                                            toDevice=False)),
                'query': DataLoader(dataset=self.meta_datasets[project]['query'], batch_size=config.query_batch_size, shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab,
                                                                                            toDevice=False))
            }
        
        self.meta_dataloaders[validating_project] = {
                'support': DataLoader(dataset=self.meta_datasets[validating_project]['support'], batch_size=config.support_batch_size, shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab,
                                                                                            toDevice=False)),
                'query': DataLoader(dataset=self.meta_datasets[validating_project]['query'], batch_size=config.query_batch_size, shuffle=True,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab,
                                                                                            toDevice=False))
            }
        # vocab
        self.code_vocab: utils.Vocab
        self.ast_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.ast_vocab = utils.load_vocab_pk(ast_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
        # new vocab
        # else:
        #     self.code_vocab = utils.Vocab('code_vocab')
        #     self.ast_vocab = utils.Vocab('ast_vocab')
        #     self.nl_vocab = utils.Vocab('nl_vocab')
        #     codes, asts, nls = self.meta_datasets.get_dataset()
        #     for code, ast, nl in zip(codes, asts, nls):
        #         self.code_vocab.add_sentence(code)
        #         self.ast_vocab.add_sentence(ast)
        #         self.nl_vocab.add_sentence(nl)

        #     self.origin_code_vocab_size = len(self.code_vocab)
        #     self.origin_nl_vocab_size = len(self.nl_vocab)

        #     # trim vocabulary
        #     self.code_vocab.trim(config.code_vocab_size)
        #     self.nl_vocab.trim(config.nl_vocab_size)
        #     # save vocabulary
        #     self.code_vocab.save(config.code_vocab_path)
        #     self.ast_vocab.save(config.ast_vocab_path)
        #     self.nl_vocab.save(config.nl_vocab_path)
        #     self.code_vocab.save_txt(config.code_vocab_txt_path)
        #     self.ast_vocab.save_txt(config.ast_vocab_txt_path)
        #     self.nl_vocab.save_txt(config.nl_vocab_txt_path)

        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab_size = len(self.nl_vocab)

        # model
        self.model = models.Model(code_vocab_size=self.code_vocab_size,
                            ast_vocab_size=self.ast_vocab_size,
                            nl_vocab_size=self.nl_vocab_size,
                            model_file_path=model_file_path)
        self.maml=l2l.algorithms.MAML(self.model, lr=0.05)
        # self.params = list(self.model.module.code_encoder.parameters()) + \
        #     list(self.model.module.ast_encoder.parameters()) + \
        #     list(self.model.module.reduce_hidden.parameters()) + \
        #     list(self.model.module.decoder.parameters())
        
        # optimizer
        # self.optimizer = Adam([
        #     {'params': self.model.module.code_encoder.parameters(), 'lr': config.code_encoder_lr},
        #     {'params': self.model.module.ast_encoder.parameters(), 'lr': config.ast_encoder_lr},
        #     {'params': self.model.module.reduce_hidden.parameters(), 'lr': config.reduce_hidden_lr},
        #     {'params': self.model.module.decoder.parameters(), 'lr': config.decoder_lr},
        # ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.params=self.maml.parameters()
        self.optimizer=Adam(self.maml.parameters(),lr=config.learning_rate)
        self.eval_instance = eval.Eval(self.get_cur_state_dict(),code_path=f'../dataset_v2/original/{validating_project}/valid_transfer.code',ast_path=f'../dataset_v2/original/{validating_project}/valid_transfer.sbt',nl_path=f'../dataset_v2/original/{validating_project}/valid_transfer.comment')

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)

        # best score and model(state dict)
        self.min_loss: float = 1000
        self.best_model: dict = {}
        self.best_epoch_batch: (int, int) = (None, None)

        # eval instance
        # self.eval_instance = eval.Eval(self.get_cur_state_dict())

        # early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            self.early_stopping = utils.EarlyStopping()

        config.model_dir = os.path.join(config.model_dir, utils.get_timestamp())
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)

    def run_train(self):
        """
        start training
        """
        with torch.backends.cudnn.flags(enabled=False):
            self.train_iter()
            return self.best_model

    def run_one_batch(self, model, batch, batch_size, criterion):
        """
        train one batch
        :param batch: get from collate_fn of corresponding dataloader
        :param batch_size: batch size
        :param criterion: loss function
        :return: avg loss
        """
        model.train()
        nl_batch = batch[4]

        decoder_outputs = model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

        decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
        nl_batch = nl_batch.view(-1)

        loss = criterion(decoder_outputs, nl_batch)
        return loss

    def eval_one_batch(self, model, batch, batch_size, criterion):
        """
        train one batch
        :param batch: get from collate_fn of corresponding dataloader
        :param batch_size: batch size
        :param criterion: loss function
        :return: avg loss
        """
        model.eval()
        with torch.no_grad():
            nl_batch = batch[4]

            decoder_outputs = model(batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

            decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
            nl_batch = nl_batch.view(-1)

            loss = criterion(decoder_outputs, nl_batch)

            return loss

    def train_iter(self,train_steps=12000, inner_train_steps=2, 
              valid_steps=200, inner_valid_steps=4, 
              valid_every=5, eval_start=0, early_stop=50, epoch_number=12000):

        self.criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        #self.maml = l2l.algorithms.MAML(self.model, lr=0.1, allow_nograd=True)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # if config.use_lr_decay:
        #     self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
        #                                             step_size=config.lr_decay_every,
        #                                             gamma=config.lr_decay_rate)
        
        for epoch in range(epoch_number):
            # support_iterators = {project: iter(self.meta_dataloaders[project]['support']) for project in self.training_projects}
            # query_iterators = {project: iter(self.meta_dataloaders[project]['query']) for project in self.training_projects}
            # num_iteration=max([len(self.meta_dataloaders[project]['support']) for project in self.training_projects ])
            #print(f'[DEBUG] Num iteration: {num_iteration} \n')
            num_iteration=50
            pbar = tqdm(range(num_iteration))
            idx=0

            for iteration in pbar: # outer loop
                #projects=random.sample(self.training_projects, 4)
                losses = []
                self.optimizer.zero_grad() 
                for project in self.training_projects: # inner loop
                    sup_iter=iter(self.meta_dataloaders[project]['support'])
                    sup_batch = next(sup_iter) 
                    qry_batch = next(sup_iter)
                    sup_batch, qry_batch=tuple_map(lambda x: x.to(config.device) if type(x) is torch.Tensor else x,(sup_batch, qry_batch))

                    # try:
                    #     sup_batch = next(support_iterators[project])
                    #     qry_batch = next(query_iterators[project])
                    # except StopIteration:
                    #     # Reset iterators if StopIteration is encountered
                    #     support_iterators[project] = iter(self.meta_dataloaders[project]['support'])
                    #     query_iterators[project] = iter(self.meta_dataloaders[project]['query'])
                    #     sup_batch = next(support_iterators[project])
                    #     qry_batch = next(query_iterators[project])
                    batch_size_sup = len(sup_batch[0][0])
                    batch_size_qry = len(qry_batch[0][0])
                    #print(f'[DEBUG] Batch size sup: {batch_size_sup}, Batch size query: {batch_size_qry} \n')

                    task_model = self.maml.clone()
                    for _ in range(inner_train_steps):
                        adaptation_loss=self.run_one_batch(task_model,sup_batch,batch_size_sup,self.criterion)
                        task_model.adapt(adaptation_loss) 
                    query_loss=self.run_one_batch(task_model,qry_batch,batch_size_qry,self.criterion)
                    query_loss.backward()
                    losses.append(query_loss.item())

                torch.nn.utils.clip_grad_norm_(self.params, 5)
                self.optimizer.step()
                pbar.set_description('Epoch = %d, iteration = %d, [loss=%.4f, min=%.4f, max=%.4f] \n' % (epoch, idx, np.mean(losses), np.min(losses), np.max(losses)))
                config.logger.info('epoch: {}/{}, iteration: {}/{}, avg loss: {:.4f}'.format(
                        epoch + 1, epoch_number,iteration+1 , num_iteration, np.mean(losses)))
                idx+=1

            if config.use_lr_decay:
                self.lr_scheduler.step()
                
            # validation
            if epoch >= eval_start:
                self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=1)
                if config.use_early_stopping:
                    if self.early_stopping.early_stop:
                        break

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}.pt'.format(
                self.best_epoch_batch[0])
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):
        """
        save current model
        :param name: if given, name the model file by given name, else by current time
        :param state_dict: if given, save the given state dict, else save current model
        :return:
        """
        if state_dict is None:
            state_dict = self.get_cur_state_dict()
        if name is None:
            model_save_path = os.path.join(config.model_dir, 'meta_model_{}.pt'.format(utils.get_timestamp()))
        else:
            model_save_path = os.path.join(config.model_dir, name)
        torch.save(state_dict, model_save_path)

    def save_check_point(self):
        pass

    def get_cur_state_dict(self) -> dict:
        """
        get current state dict of model
        :return:
        """
        state_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        return state_dict

    def valid_state_dict(self, state_dict, epoch, batch=-1):
        # Clone for valid task
        # self.eval_instance.set_state_dict(state_dict["model"])
        # loss = self.eval_instance.run_eval()
        # adapt
        # dataset_dir = "../dataset_v2/"
        # train_instance = train.Train(vocab_file_path=self.vocab_file_path, model_state_dict=state_dict,
        #                             code_path=os.path.join(dataset_dir,f'original/{self.validating_project}/train.code')
        #                             ,ast_path=os.path.join(dataset_dir,f'original/{self.validating_project}/train.sbt'),
        #                             nl_path=os.path.join(dataset_dir,f'original/{self.validating_project}/train.comment'),batch_size=config.support_batch_size,
        #                             code_valid_path=f'../dataset_v2/original/{self.validating_project}/valid_transfer.code',nl_valid_path=f'../dataset_v2/original/{self.validating_project}/valid_transfer.comment',
        #                                 ast_valid_path=f'../dataset_v2/original/{self.validating_project}/valid_transfer.sbt'
        #                                 ,save_file=False)
        # best_model_test_dict=train_instance.run_train()
        # eval_instance = eval.Eval(best_model_test_dict,code_path=os.path.join(dataset_dir,f'original/{self.validating_project}/valid.code')
        #                         ,ast_path=os.path.join(dataset_dir,f'original/{self.validating_project}/valid.sbt'),
        #                         nl_path=os.path.join(dataset_dir,f'original/{self.validating_project}/valid.comment'))
        # loss = eval_instance.run_eval()
        losses = []
        for batch_s,batch_q in zip(self.meta_dataloaders[self.validating_project]['support'],self.meta_dataloaders[self.validating_project]['query']):
            task_model = self.maml.clone()
            batch_sc,batch_qc=tuple_map(lambda x: x.to(config.device) if type(x) is torch.Tensor else x,(batch_s,batch_q))
            adaptation_loss=self.run_one_batch(task_model,batch_sc,len(batch_s[0][0]),self.criterion)
            task_model.adapt(adaptation_loss)
            losses.append(self.eval_one_batch(task_model,batch_qc,len(batch_q[0][0]),self.criterion).item())

        loss = sum(losses)/len(losses)
        print("Validation complete for epoch ",epoch," with average loss: ",loss)
        config.logger.info(f'Validation complete for epoch {epoch} with average loss: {loss}')
        if config.save_valid_model:
            model_name = 'meta_model_valid-loss-{:.4f}_epoch-{}_batch-{}.pt'.format(loss, epoch, batch)
            save_thread = threading.Thread(target=self.save_model, args=(model_name, state_dict))
            save_thread.start()

        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)

        if config.use_early_stopping:
            self.early_stopping(loss)

