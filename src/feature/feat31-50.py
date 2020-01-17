#!/usr/bin/env python
#coding=utf-8

# 生成词向量距离特征

# 基础模块
import os
import gc
import sys
import time
import pickle
from datetime import datetime
from tqdm import tqdm

# 数据处理
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

# 自定义工具包
sys.path.append('../../tools/')
import loader
import pandas_util
import custom_bm25 as bm25
from feat_utils import try_divide, dump_feat_name

# 开源工具包
import nltk
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# 设置随机种子
SEED = 2020

input_root_path  = '../../input/'
output_root_path = '../../feat/'

postfix = '31-50'
file_type = 'ftr'

# 当前特征
tr_fea_out_path = output_root_path + 'tr_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = output_root_path + 'te_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_s0_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_s0_{}.{}'.format(postfix, file_type)

ID_NAMES = ['description_id', 'paper_id']
PROCESS_NUM = 15

# load data
ts = time.time()
dictionary = corpora.Dictionary.load('../../feat/corpus.dict')
tfidf = models.TfidfModel.load('../../feat/tfidf.model')

print ('load data completed, cost {}s'.format(np.round(time.time() - ts, 2)))
                               
def sum_score(x, y):
    return max(x, 0) + max(y, 0)

def cos_dis(vec_x, vec_y, norm=False):
    if vec_x == None or vec_y == None:
        return -1
    dic_x = {v[0]: v[1] for v in vec_x}
    dic_y = {v[0]: v[1] for v in vec_y}
    
    dot_prod = 0
    for k, x in dic_x.items():
        y = dic_y.get(k, 0)
        dot_prod += x * y
    norm_x = np.linalg.norm([v[1] for v in vec_x]) 
    norm_y = np.linalg.norm([v[1] for v in vec_y])
    
    cos = dot_prod / (norm_x * norm_y)
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

def eucl_dis(vec_x, vec_y):
    if vec_x == None or vec_y == None:
        return -1
    dic_x = {v[0]: v[1] for v in vec_x}
    dic_y = {v[0]: v[1] for v in vec_y}
    lis_i = list(set(list(dic_x.keys()) + list(dic_y.keys())))
    squa_sum = 0
    for i in lis_i:
        x, y = dic_x.get(i, 0), dic_y.get(i, 0)
        squa_sum += np.square(x - y)
    return np.sqrt(squa_sum)

def manh_dis(vec_x, vec_y):
    if vec_x == None or vec_y == None:
        return -1
    dic_x = {v[0]: v[1] for v in vec_x}
    dic_y = {v[0]: v[1] for v in vec_y}
    lis_i = list(set(list(dic_x.keys()) + list(dic_y.keys())))
    abs_sum = 0
    for i in lis_i:
        x, y = dic_x.get(i, 0), dic_y.get(i, 0)
        abs_sum += np.abs(x - y)
    return abs_sum    

def get_bm25_corp(quer, paper_id):
    quer_vec = dictionary.doc2bow(quer.split(' '))
    corp_score = bm25_corp.get_score(quer_vec, paper_ids.index(paper_id))
    return corp_score

def get_bm25_abst(quer, paper_id):
    quer_vec = dictionary.doc2bow(quer.split(' '))
    abst_score = bm25_abst.get_score(quer_vec, paper_ids.index(paper_id))
    return abst_score

def get_bm25_titl(quer, paper_id):
    quer_vec = dictionary.doc2bow(quer.split(' '))
    titl_score = bm25_titl.get_score(quer_vec, paper_ids.index(paper_id))
    return titl_score

def single_process_feat(params=None):
    ts = time.time()
    (df, i) = params
    
    ts = time.time()
    print (i, 'start', datetime.now())
    # tfidf vec dis
    df['quer_key_vec'] = df['quer_key'].progress_apply(lambda s: tfidf[dictionary.doc2bow(s.split(' '))])
    df['quer_all_vec'] = df['quer_all'].progress_apply(lambda s: tfidf[dictionary.doc2bow(s.split(' '))])
    df['titl_vec'] = df['titl'].progress_apply(lambda s: tfidf[dictionary.doc2bow(s.split(' '))])
    df['abst_vec'] = df['abst'].progress_apply(lambda s: tfidf[dictionary.doc2bow(s.split(' '))])
    df['corp_vec'] = df['corp'].progress_apply(lambda s: tfidf[dictionary.doc2bow(s.split(' '))])                               
    print (i, 'load vec completed, cost {}s'.format(np.round(time.time() - ts), 2))
    
    ts = time.time()
    vec_type = 'tfidf'
    for vec_x in ['quer_key', 'quer_all']:
        for vec_y in ['abst', 'titl', 'corp']:
            df['{}_{}_{}_cos_dis'.format(vec_x, vec_type, vec_y)] = df.progress_apply(lambda row: \
                cos_dis(row['{}_vec'.format(vec_x)], row['{}_vec'.format(vec_y)]), axis=1)
            df['{}_{}_{}_eucl_dis'.format(vec_x, vec_type, vec_y)] = df.progress_apply(lambda row: \
                eucl_dis(row['{}_vec'.format(vec_x)], row['{}_vec'.format(vec_y)]), axis=1) 
            df['{}_{}_{}_manh_dis'.format(vec_x, vec_type, vec_y)] = df.progress_apply(lambda row: \
                manh_dis(row['{}_vec'.format(vec_x)], row['{}_vec'.format(vec_y)]), axis=1) 
            
        print (i, vec_x, 'tfidf completed, cost {}s'.format(np.round(time.time() - ts), 2))
    
    del_cols = [col for col in df.columns if df[col].dtype == 'O' and col not in ID_NAMES]
    print ('del cols', del_cols)
    df.drop(del_cols, axis=1, inplace=True)
    return df

def partition(df, num):
    df_partitions, step = [], int(np.ceil(df.shape[0]/num))
    for i in range(0, df.shape[0], step):
        df_partitions.append(df.iloc[i:i+step])
    return df_partitions

def multi_process_feat(df):
    pool = Pool(PROCESS_NUM)
    df = df[ID_NAMES + ['quer_key', 'quer_all', 'abst', 'titl', 'corp']]
    df_parts = partition(df, PROCESS_NUM)
    print ('{} processes init and partition to {} parts' \
           .format(PROCESS_NUM, PROCESS_NUM))
    ts = time.time()

    param_list = [(df_parts[i], i) \
            for i in range(PROCESS_NUM)]
    dfs = pool.map(single_process_feat, param_list)
    df_out = pd.concat(dfs, axis=0)
    return df_out

def gen_samples(paper, tr_desc_path, tr_recall_path, fea_out_path):
    tr_desc = loader.load_df(tr_desc_path)
    tr = loader.load_df(tr_recall_path)
#     tr = tr.head(1000)
    
    tr = tr.merge(paper, on=['paper_id'], how='left')
    tr = tr.merge(tr_desc[['description_id', 'quer_key', 'quer_all']], on=['description_id'], how='left')

    print (tr.columns)
    print (tr.head())
    
    tr_feat = multi_process_feat(tr)
    loader.save_df(tr_feat, fea_out_path)
    
    tr = tr.merge(tr_feat, on=ID_NAMES, how='left')
    del_cols = [col for col in tr.columns if tr[col].dtype == 'O' and col not in ID_NAMES]
    print ('tr del cols', del_cols)
    return tr.drop(del_cols, axis=1)


# 增加 vec sim 特征

if __name__ == "__main__":

    ts = time.time()
    tqdm.pandas()
    print('start time: %s' % datetime.now())
    paper = loader.load_df('../../input/paper_input_final.ftr')
    paper['abst'] = paper['abst'].apply(lambda s: s.replace('no_content', ''))
    paper['corp'] = paper['abst'] + ' ' + paper['titl'] + ' ' + paper['keywords'].fillna('').replace(';', ' ')
    
    tr_desc_path = '../../input/tr_input_final.ftr'
    te_desc_path = '../../input/te_input_final.ftr'
    
    tr_recall_path = '../../feat/tr_s0_30-50.ftr'
    te_recall_path = '../../feat/te_s0_30-50.ftr'
    
    tr = gen_samples(paper, tr_desc_path, tr_recall_path, tr_fea_out_path)
    print (tr.columns)
    print ([col for col in tr.columns if tr[col].dtype == 'O'])
    loader.save_df(tr, tr_out_path)
    
    te = gen_samples(paper, te_desc_path, te_recall_path, te_fea_out_path)
    print (te.columns)
    loader.save_df(te, te_out_path)
    print('all completed: {}, cost {}s'.format(datetime.now(), np.round(time.time() - ts, 2)))




