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
from preprocess import preprocess
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

FEA_NUM = 38

postfix = 's0_{}'.format(FEA_NUM)
file_type = 'ftr'

# 当前特征
tr_fea_out_path = output_root_path + 'tr_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = output_root_path + 'te_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_{}.{}'.format(postfix, file_type)

ID_NAMES = ['description_id', 'paper_id']
PROCESS_NUM = 20

# load data
ts = time.time()

def feat_extract(tr_path, te_path, prefix):
    tr_sample = loader.load_df('../../feat/tr_s0_37.ftr')
    te_sample = loader.load_df('../../feat/te_s0_37.ftr')
   
    tr = loader.load_df(tr_path)
    te = loader.load_df(te_path)
    
    del_cols = ['label']
    del_cols = [col for col in tr.columns if col in del_cols]
    tr.drop(del_cols, axis=1, inplace=True)

    tr = tr_sample[ID_NAMES].merge(tr, on=ID_NAMES, how='left')
    te = te_sample[ID_NAMES].merge(te, on=ID_NAMES, how='left')

    tr.columns = ID_NAMES + [prefix]
    te.columns = ID_NAMES + [prefix]

    print (prefix)
    print (tr.shape, te.shape)
    print (tr.head())
    
    tr = tr[prefix]
    te = te[prefix]

    return tr, te

def output_fea(tr, te):
    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_fea_out_path)
    loader.save_df(te, te_fea_out_path)

# 生成特征
def gen_fea(base_tr_path=None, base_te_path=None):

    tr_sample = loader.load_df('../../feat/tr_s0_37.ftr')
    te_sample = loader.load_df('../../feat/te_s0_37.ftr')

    prefixs = ['m1_cat_03', 'm1_infesent_simple', 'm1_nn_02', \
               'm2_ESIM_001', 'm2_ESIMplus_001', 'lgb_m3_37-0']
    
    tr_paths = ['{}_tr.ftr'.format(prefix) for prefix in prefixs]
    te_paths = ['final_{}_te.ftr'.format(prefix) for prefix in prefixs]
    
    tr_paths = ['../../stk_feat/{}'.format(p) for p in tr_paths]
    te_paths = ['../../stk_feat/{}'.format(p) for p in te_paths]


    trs, tes = [], []
    for i, prefix in enumerate(prefixs):
        tr, te = feat_extract(tr_paths[i], te_paths[i], prefix + '_prob')
        trs.append(tr)
        tes.append(te)
    tr = pd.concat([tr_sample[ID_NAMES]] + trs, axis=1)
    te = pd.concat([te_sample[ID_NAMES]] + tes, axis=1)

    float_cols = [c for c in tr.columns if tr[c].dtype == 'float']
    tr[float_cols] = tr[float_cols].astype('float32')
    te[float_cols] = te[float_cols].astype('float32')

    print (tr.shape, te.shape)
    print (tr.head())
    print (te.head())
    print (tr.columns)

    output_fea(tr, te)

# merge 已有特征
def merge_fea(tr_list, te_list):
    tr = loader.merge_fea(tr_list, primary_keys=ID_NAMES)
    te = loader.merge_fea(te_list, primary_keys=ID_NAMES)

    print (tr.head())
    print (te.head())
    print (tr.columns.tolist())

    loader.save_df(tr, tr_out_path)
    loader.save_df(te, te_out_path)
    
if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../../feat/'
    base_tr_path = root_path + 'tr_s0_37.ftr'
    base_te_path = root_path + 'te_s0_37.ftr'

    gen_fea()

    # merge fea
    prefix = 's0'
    fea_list = [FEA_NUM]

    tr_list = [base_tr_path] + \
            [root_path + 'tr_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]
    te_list = [base_te_path] + \
            [root_path + 'te_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]

    merge_fea(tr_list, te_list)

    print('all completed: %s' % datetime.now())



