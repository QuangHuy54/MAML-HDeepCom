import torch
import torch.backends.cudnn as cudnn
import data
import train
import argparse
import os
import utils
import config
import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import models

#parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
#parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
#parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
#parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')
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
            results_1 = torch.cat(results_1,1).cpu().numpy()
            results_2 = torch.cat(results_2,1).cpu().numpy()
        return results_1,results_2
class ARGS:
  save_dir='/result_tsne/'
  seed=1
  batch_size=16
args = ARGS()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

vocab_file_path=(config.code_vocab_path, config.ast_vocab_path, config.nl_vocab_path)
code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
code_vocab = utils.load_vocab_pk(code_vocab_path)
ast_vocab = utils.load_vocab_pk(ast_vocab_path)
nl_vocab = utils.load_vocab_pk(nl_vocab_path)
code_vocab_size = len(code_vocab)
ast_vocab_size = len(ast_vocab)
nl_vocab_size = len(nl_vocab)
dataset_dir = "../dataset_v2/"
dataset_springf = data.CodePtrDataset(os.path.join(dataset_dir,f'original/spring-framework/all_truncated_final.code'),
                                    os.path.join(dataset_dir,f'original/spring-framework/all_truncated.sbt'),
                                    os.path.join(dataset_dir,f'original/spring-framework/all_truncated_final.comment'))
dataset_springb=data.CodePtrDataset(os.path.join(dataset_dir,f'original/spring-boot/all_truncated_final.code'),
                                    os.path.join(dataset_dir,f'original/spring-boot/all_truncated.sbt'),
                                    os.path.join(dataset_dir,f'original/spring-boot/all_truncated_final.comment'))
dataset_flink=data.CodePtrDataset(os.path.join(dataset_dir,f'original/dubbo/all_truncated_final.code'),
                                    os.path.join(dataset_dir,f'original/dubbo/all_truncated.sbt'),
                                    os.path.join(dataset_dir,f'original/dubbo/all_truncated_final.comment'))

dataloader_springf = torch.utils.data.DataLoader(dataset_springf, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                                code_vocab=code_vocab,
                                                                                                ast_vocab=ast_vocab,
                                                                                                nl_vocab=nl_vocab,
                                                                                                size1=360,size2=1005))
dataloader_springb = torch.utils.data.DataLoader(dataset_springb, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                                code_vocab=code_vocab,
                                                                                                ast_vocab=ast_vocab,
                                                                                                nl_vocab=nl_vocab,
                                                                                                size1=360,size2=1005))
dataloader_flink = torch.utils.data.DataLoader(dataset_flink, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                                code_vocab=code_vocab,
                                                                                                ast_vocab=ast_vocab,
                                                                                                nl_vocab=nl_vocab,
                                                                                                size1=360,size2=1005))
# set model
train_instance = train.Train(vocab_file_path=(config.code_vocab_path, config.ast_vocab_path, config.nl_vocab_path),model_file_path='model/spring-security_meta_3/best_epoch-0.pt',
                                  code_path=os.path.join(dataset_dir,f'original/spring-security/fold_0_train.code')
                                  ,ast_path=os.path.join(dataset_dir,f'original/spring-security/fold_0_train.sbt'),
                                  nl_path=os.path.join(dataset_dir,f'original/spring-security/fold_0_train.comment'),batch_size=config.support_batch_size,
                                  code_valid_path=os.path.join(dataset_dir,f'original/kafka/all_truncated_final.code'),nl_valid_path=os.path.join(dataset_dir,f'original/kafka/all_truncated_final.comment'),
                                        ast_valid_path=os.path.join(dataset_dir,f'original/kafka/all_truncated.sbt')
                                        ,num_of_data=100,save_file=False,seed=0,is_test=True)
best_model_test_dict=train_instance.run_train()

del train_instance
torch.cuda.empty_cache()

dataset_springs=data.CodePtrDataset(os.path.join(dataset_dir,f'original/dubbo/fold_0_train.code'),
                                    os.path.join(dataset_dir,f'original/dubbo/fold_0_train.sbt'),
                                    os.path.join(dataset_dir,f'original/dubbo/fold_0_train.comment'),100,0)
dataloader_springs = torch.utils.data.DataLoader(dataset_springs, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                                code_vocab=code_vocab,
                                                                                                ast_vocab=ast_vocab,
                                                                                                nl_vocab=nl_vocab,
                                                                                                size1=360,size2=1005))
model=models.Model(code_vocab_size=code_vocab_size,
                              ast_vocab_size=ast_vocab_size,
                              nl_vocab_size=nl_vocab_size,
                              model_state_dict=best_model_test_dict
                              ,is_eval=True)
doc2vec = RNNDoc2Vec(model)
def gen_features():
    targets_list = []
    outputs_list = []

    code_output_springf,ast_output_springf = doc2vec.get_vecs(dataloader_springf)
    target_springf=np.full((len(dataset_springf), 1), 0)

    targets_list.append(target_springf)
    outputs_list.append(code_output_springf)

    code_output_springb,ast_output_springb = doc2vec.get_vecs(dataloader_springb)
    target_springb=np.full((len(dataset_springb), 1), 1)
    
    targets_list.append(target_springb)
    outputs_list.append(code_output_springb)

    code_output_flink,ast_output_flink = doc2vec.get_vecs(dataloader_flink)
    target_flink=np.full((len(dataset_flink), 1), 2)
    
    targets_list.append(target_flink)
    outputs_list.append(code_output_flink)

    code_output_springs,ast_output_springs = doc2vec.get_vecs(dataloader_springs)
    target_springs=np.full((len(dataset_springs), 1), 3)
    
    targets_list.append(target_springs)
    outputs_list.append(code_output_springs)

    targets = np.concatenate(targets_list, axis=1)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    targets=np.swapaxes(targets,0,1)
    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne.png'), bbox_inches='tight')
    print('done!')

targets, outputs = gen_features()
tsne_plot(args.save_dir, targets, outputs)