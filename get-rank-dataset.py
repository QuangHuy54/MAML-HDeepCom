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
import multiprocessing 
import sys
import builtins

def get_rouge(data_1,data_2,q=None):
    evaluator = Rouge(metrics=['rouge-n'],max_n=2,
                        apply_avg=False,
                        apply_best=True)
    total_score=0
    for sentence in data_1:
        scores = evaluator.get_scores([sentence], [data_2])
        total_score+=scores['rouge-2']['r']
    total_score/=len(data_1)
    if q!=None:
        q.put(total_score)
    return total_score

def get_rouge_core(data_1,data_2,start,end,q):
    evaluator = Rouge(metrics=['rouge-n'],max_n=2,
                        apply_avg=False,
                        apply_best=True)
    total_score=0
    for i in range(start,end):
        scores=evaluator.get_scores([data_1[i]], [data_2])
        total_score+=scores['rouge-2']['r']
    q.put(total_score)     
def get_total_length(data):
    total_length=0
    for sentence in data:
        total_length+=len(sentence.split())
    return total_length
def list_of_strings(arg):
    return arg.split(',')

def get_result(project,projects):
    rank_code={}
    rank_rouge={}
    rank_length={}
    rank_ast={}
    with open(f'../dataset_v2/original/{project}/all_truncated_final.code',"r") as f1:
        data_original=f1.readlines()
    original_code_semantic=torch.load(os.path.join(dataset_dir,f'{project}/all_code_semantic.pt'), map_location='cpu')
    original_code_semantic=torch.max(original_code_semantic, 0).values
    original_ast_semantic=torch.load(os.path.join(dataset_dir,f'{project}/all_ast_semantic.pt'), map_location='cpu')
    original_ast_semantic=torch.max(original_ast_semantic, 0).values
    #print(projects)
    for target_pro in projects:
        target_semantic=torch.load(os.path.join(dataset_dir,f'{target_pro}/all_code_semantic.pt'), map_location='cpu')
        target_semantic=torch.max(target_semantic, 0).values
        target_ast_semantic=torch.load(os.path.join(dataset_dir,f'{target_pro}/all_ast_semantic.pt'), map_location='cpu')
        target_ast_semantic=torch.max(target_ast_semantic, 0).values
        rank_code[target_pro]=torch.cosine_similarity(original_code_semantic,target_semantic,-1).item()
        rank_ast[target_pro]=torch.cosine_similarity(original_ast_semantic,target_ast_semantic,-1).item()
        # with open(f'../dataset_v2/original/{target_pro}/all_truncated_final.code',"r") as f2:
        #     data_target=f2.readlines()
        # rank_rouge[target_pro]=get_rouge(data_original,data_target)
        # rank_length[target_pro]=abs(get_total_length(data_original)-get_total_length(data_target))
    #print(rank_code)  
    rank_code=dict(sorted(rank_code.items(), key=lambda item: item[1],reverse=True))
    rank_ast=dict(sorted(rank_ast.items(), key=lambda item: item[1],reverse=True))
    # rank_rouge=dict(sorted(rank_rouge.items(), key=lambda item: item[1],reverse=True))
    # rank_length=dict(sorted(rank_length.items(), key=lambda item: item[1]))
    with open(f'../dataset_v2/original/{project}/result_meta_dataset.txt',"w") as f:
        f.write(project+'\n')
        f.flush()
        print(project, flush=True)
        print("Rank code: ",' '.join([x for x in rank_code.keys()]), flush=True)
        # print("Rank rouge: ",' '.join([x for x in rank_rouge.keys()]), flush=True)
        # print("Rank length: ",' '.join([x for x in rank_length.keys()]), flush=True)
        print("Rank ast: ",' '.join([x for x in rank_ast.keys()]), flush=True)
        f.write("Rank code: "+' '.join([x for x in rank_code.keys()])+'\n')
        f.flush()
        # f.write("Rank rouge: "+' '.join([x for x in rank_rouge.keys()])+'\n')
        # f.flush()
        # f.write("Rank length: "+' '.join([x for x in rank_length.keys()])+'\n')
        # f.flush()
        f.write("Rank ast: "+' '.join([x for x in rank_ast.keys()])+'\n')
        f.flush()
        ranking={}
        ranking_new={}
        for idx,k in enumerate(rank_code):
            ranking[k]=idx+1
            ranking_new[k]=idx+1
        # for idx,k in enumerate(rank_rouge):
        #     ranking[k]=ranking[k]+idx+1
        #     ranking_new[k]=ranking_new[k]+idx+1
        # for idx,k in enumerate(rank_length):
        #     ranking[k]=ranking[k]+idx+1
        #    ranking_new[k]=ranking_new[k]+idx+1
        for idx,k in enumerate(rank_ast):
            ranking_new[k]=ranking_new[k]+idx+1
        for key in ranking.keys():
            ranking[key]=float(ranking[key])/1
        for key in ranking_new.keys():
            ranking_new[key]=float(ranking_new[key])/2
        ranking=dict(sorted(ranking.items(), key=lambda item: item[1]))
        ranking_new=dict(sorted(ranking_new.items(), key=lambda item: item[1]))
        top_4_result=[]
        top_4_result_new=[]
        for idx,key in enumerate(ranking):
            top_4_result.append(key)
            if idx==3:
                break
        for idx,key in enumerate(ranking_new):
            top_4_result_new.append(key)
            if idx==3:
                break
        print("Top 4 (not include ast): ",' '.join([x for x in top_4_result]), flush=True)
        f.write("Top 4 (not include ast): "+' '.join([x for x in top_4_result])+'\n')
        f.flush()
        print("Top 4: ",' '.join([x for x in top_4_result_new]), flush=True)
        f.write("Top 4: "+' '.join([x for x in top_4_result_new])+'\n')
        f.flush()  

if __name__ == '__main__':
    original_projects=['dagger','dubbo','ExoPlayer','flink','guava','kafka','spring-boot','spring-framework','spring-security']
    dataset_dir = "../dataset_v2/original/"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--project', type=list_of_strings,default=None)
    parser.add_argument('-c', '--core', type=int,default=None)
    args = parser.parse_args()
    test_projects=args.project
    num_core=args.core
    list_process=[]        
    for i in range(len(test_projects)):
        project=test_projects[i]
        projects=original_projects.copy()
        projects.remove(project)
        p=multiprocessing.Process(target=get_result,args=(project,projects))
        print(f"Run process {i}", flush=True)
        p.start()
        list_process.append(p)

    for i in range(len(test_projects)):
        list_process[i].join() 
        print(f"Finish process {i}", flush=True)   
    # with open(f'../dataset_v2/original/result_meta_dataset.txt',"w") as f:
    #     for project in test_projects:
    #         projects=original_projects.copy()
    #         projects.remove(project)
    #         rank_code={}
    #         rank_rouge={}
    #         rank_length={}
    #         with open(f'../dataset_v2/original/{project}/all_truncated_final.code',"r") as f1:
    #             data_original=f1.readlines()
    #         original_code_semantic=torch.load(os.path.join(dataset_dir,f'{project}/all_code_semantic.pt'), map_location='cpu')
    #         original_code_semantic=torch.max(original_code_semantic, 0).values
    #         #print(projects)
    #         for target_pro in projects:
    #             target_semantic=torch.load(os.path.join(dataset_dir,f'{target_pro}/all_code_semantic.pt'), map_location='cpu')
    #             target_semantic=torch.max(target_semantic, 0).values
    #             rank_code[target_pro]=torch.cosine_similarity(original_code_semantic,target_semantic,-1).item()
    #             with open(f'../dataset_v2/original/{target_pro}/all_truncated_final.code',"r") as f2:
    #                 data_target=f2.readlines()
    #             rank_rouge[target_pro]=get_rouge(data_original,data_target)
    #             rank_length[target_pro]=abs(get_total_length(data_original)-get_total_length(data_target))
    #         #print(rank_code)  
    #         rank_code=dict(sorted(rank_code.items(), key=lambda item: item[1],reverse=True))
    #         rank_rouge=dict(sorted(rank_rouge.items(), key=lambda item: item[1],reverse=True))
    #         rank_length=dict(sorted(rank_length.items(), key=lambda item: item[1]))
    #         f.write(project+'\n')
    #         print(project)
    #         print("Rank code: ",' '.join([x for x in rank_code.keys()]))
    #         print("Rank rouge: ",' '.join([x for x in rank_rouge.keys()]))
    #         print("Rank length: ",' '.join([x for x in rank_length.keys()]))
    #         f.write("Rank code: "+' '.join([x for x in rank_code.keys()])+'\n')
    #         f.write("Rank rouge: "+' '.join([x for x in rank_rouge.keys()])+'\n')
    #         f.write("Rank length: "+' '.join([x for x in rank_length.keys()])+'\n')
    #         ranking={}
    #         for idx,k in enumerate(rank_code):
    #             ranking[k]=idx+1
    #         for idx,k in enumerate(rank_rouge):
    #             ranking[k]=ranking[k]+idx+1
    #         for idx,k in enumerate(rank_length):
    #             ranking[k]=ranking[k]+idx+1
    #         for key in ranking.keys():
    #             ranking[key]=float(ranking[key])/3
    #         ranking=dict(sorted(ranking.items(), key=lambda item: item[1]))
    #         top_4_result=[]
    #         for idx,key in enumerate(ranking):
    #             top_4_result.append(key)
    #             if idx==3:
    #                 break
    #         print("Top 4: ",' '.join([x for x in top_4_result]))
    #         f.write("Top 4: "+' '.join([x for x in top_4_result])+'\n')         