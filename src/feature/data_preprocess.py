#!/usr/bin/env python
#coding=utf-8

# 基础模块
import os
import sys
import time
from tqdm import tqdm
from datetime import datetime

# 数据处理
import re
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool

# 自定义工具包
sys.path.append('../../tools/')
import loader
import pandas_util
from nlp_preprocess import preprocess

# 设置随机种子
SEED = 2020
PROCESS_NUM, PARTITION_NUM = 32, 32

input_root_path  = '../../input/'
output_root_path = '../../input/'

postfix = 'final_all'
file_type = 'ftr'

tr_out_path = output_root_path + 'tr_input_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_input_{}.{}'.format(postfix, file_type)
paper_out_path = output_root_path + 'paper_input_{}.{}'.format(postfix, file_type)

# 获取关键句函数
def digest(text):
    backup = text[:]
    text = text.replace('al.', '').split('. ')
    t=''
    pre_text=[]
    len_text=len(text)-1
    add=True
    pre=''
    while len_text>=0:
        index=text[len_text]
        index+=pre
        if len(index.split(' '))<=3 :
            add=False
            pre=index+pre
        else:
            add=True
            pre=''
        if add:
            pre_text.append(index)
        len_text-=1
    if len(pre_text)==0:
        pre_text=text
    pre_text.reverse()
    for index in pre_text:
        if index.find('[**##**]') != -1:
            index = re.sub(r'[\[|,]+\*\*\#\#\*\*[\]|,]+','',index)
            index+='. '
            t+=index
    return t

def partition(df, num):
    df_partitions, step = [], int(np.ceil(df.shape[0]/num))
    for i in range(0, df.shape[0], step):
        df_partitions.append(df.iloc[i:i+step])
    return df_partitions

def tr_single_process(params=None):
    (tr, i) = params
    print (i, 'start', datetime.now())
    tr['quer_key'] = tr['description_text'].fillna('').progress_apply(lambda s: preprocess(digest(s)))
    tr['quer_all'] = tr['description_text'].fillna('').progress_apply(lambda s: preprocess(s))
    print (i, 'completed', datetime.now())
    return tr

def paper_single_process(params=None):
    (df, i) = params
    print (i, 'start', datetime.now())
    df['titl'] = df['title'].fillna('').progress_apply(lambda s: preprocess(s))
    df['abst'] = df['abstract'].fillna('').progress_apply(lambda s: preprocess(s))
    print (i, 'completed', datetime.now())
    return df

def multi_text_process(df, task, process_num=30):
    pool = Pool(process_num)
    df_parts = partition(df, process_num)
    print ('{} processes init and partition to {} parts' \
           .format(process_num, process_num))
    param_list = [(df_parts[i], i) for i in range(process_num)]
    if task in ['tr', 'te']:
        dfs = pool.map(tr_single_process, param_list)
    elif task in ['paper']:
        dfs = pool.map(paper_single_process, param_list)
    df = pd.concat(dfs, axis=0)
    print (task, 'multi process completed')
    print (df.columns)
    return df

if __name__ == "__main__":

    ts = time.time()
    tqdm.pandas()
    print('start time: %s' % datetime.now())
    # load data
    df = loader.load_df(input_root_path + 'candidate_paper_for_wsdm2020.ftr')
    tr = loader.load_df(input_root_path + 'train_release.csv')
    te = loader.load_df(input_root_path + 'test.csv')
    cv = loader.load_df(input_root_path + 'cv_ids_0109.csv')

    # 过滤重复数据 & 异常数据
    tr = tr[tr['description_id'].isin(cv['description_id'].tolist())]
    tr = tr[tr.description_id != '6.45E+04']

    df = df[~pd.isnull(df['paper_id'])]
    tr = tr[~pd.isnull(tr['description_id'])]
    print ('pre', te.shape)
    te = te[~pd.isnull(te['description_id'])]
    print ('post', te.shape) 
    
    #df = df.head(1000)
    #tr = tr.head(1000)
    #te = te.head(1000)

    tr = multi_text_process(tr, task='tr')
    te = multi_text_process(te, task='te')
    df = multi_text_process(df, task='paper')
    
    tr.drop(['description_text'], axis=1, inplace=True)
    te.drop(['description_text'], axis=1, inplace=True)
    df.drop(['abstract', 'title'], axis=1, inplace=True)
    print ('text preprocess completed')
    
    loader.save_df(tr, tr_out_path)
    print (tr.columns)
    print (tr.head())
    
    loader.save_df(te, te_out_path) 
    print (te.columns)
    print (te.head())
    
    loader.save_df(df, paper_out_path)
    print (df.columns)
    print (df.head())
    
    print('all completed: {}, cost {}s'.format(datetime.now(), np.round(time.time() - ts, 2)))



