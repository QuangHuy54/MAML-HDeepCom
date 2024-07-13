import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler,SGD
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt
from rouge import Rouge
import numpy as np
import utils
import config
import data
import models
import eval
torch.manual_seed(1)
import argparse

def get_rouge(data_1,data_2):
    evaluator = Rouge(max_n=2,
                        apply_avg=False,
                        apply_max=True)
    total_score=0
    for sentence in data_1:
        scores = evaluator.get_scores([sentence], [data_2])
        total_score+=scores['rouge-2']['r']
    total_score/=len(data_1)
    return total_score

def get_total_length(data):
    total_length=0
    for sentence in data:
        total_length+=len(sentence.split())
    return total_length
def list_of_strings(arg):
    return arg.split(',')
if __name__ == '__main__':
    projects=['dagger','dubbo','ExoPlayer','flink','guava','kafka','spring-boot','spring-framework','spring-security']
    dataset_dir = "../dataset_v2/original/"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--project', type=list_of_strings,default=None)
    args = parser.parse_args()
    test_projects=args.project
    with open(f'../dataset_v2/original/result_meta_dataset.txt',"w") as f:
        for project in test_projects:
            projects.remove(project)
            rank_code=dict()
            rank_rouge=dict()
            rank_length=dict()
            with open(f'../dataset_v2/original/{project}/all_truncated_final.code',"r") as f1:
                data_original=f1.readlines()
            original_code_semantic=torch.load(os.path.join(dataset_dir,f'{project}/all_code_semantic.pt'), map_location='cpu')
            original_code_semantic=torch.nn.functional.normalize(original_code_semantic)
            for target_pro in projects:
                target_semantic=torch.load(os.path.join(dataset_dir,f'{target_pro}/all_code_semantic.pt'), map_location='cpu')
                target_semantic=torch.nn.functional.normalize(target_semantic)
                rank_code[target_pro]=np.sum(np.tensordot(original_code_semantic,target_semantic))
                with open(f'../dataset_v2/original/{target_pro}/all_truncated_final.code',"r") as f2:
                    data_target=f2.readlines()
                rank_rouge[target_pro]=get_rouge(data_original,data_target)
                rank_length[target_pro]=abs(get_total_length(data_original)-get_total_length(data_target))
            rank_code=dict(sorted(rank_code.items(), key=lambda item: item[1]).reverse())
            rank_rouge=dict(sorted(rank_rouge.items(), key=lambda item: item[1]).reverse())
            rank_length=dict(sorted(rank_length.items(), key=lambda item: item[1]).reverse())
            f.write(project+'\n')
            print(project)
            print("Rank code: ",' '.join([x for x in rank_code.keys()]))
            print("Rank rouge: ",' '.join([x for x in rank_rouge.keys()]))
            print("Rank length: ",' '.join([x for x in rank_length.keys()]))
            f.write("Rank code: "+' '.join([x for x in rank_code.keys()])+'\n')
            f.write("Rank rouge: "+' '.join([x for x in rank_rouge.keys()])+'\n')
            f.write("Rank length: "+' '.join([x for x in rank_length.keys()])+'\n')
            ranking=dict()
            for idx,k in enumerate(rank_code):
                ranking[k]=idx+1
            for idx,k in enumerate(rank_rouge):
                ranking[k]=ranking[k]+idx+1
            for idx,k in enumerate(rank_length):
                ranking[k]=ranking[k]+idx+1
            for key in ranking.keys():
                ranking[key]=float(ranking[key])/3
                ranking[key]=float(ranking[key])/3
                ranking[key]=float(ranking[key])/3
            ranking=dict(sorted(rank_code.items(), key=lambda item: item[1]).reverse())
            top_4_result=[]
            for idx,key in enumerate(ranking):
                top_4_result.append(key)
                if idx==3:
                    break
            print("Top 4: ",' '.join([x for x in top_4_result]))
            f.write("Top 4: "+' '.join([x for x in top_4_result])+'\n')                

        

