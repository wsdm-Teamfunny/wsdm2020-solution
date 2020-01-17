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

FEA_NUM = '37'
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

def feat_extract(df, is_te=False):
    if is_te:
        df_pred = loader.load_df('../../output/m3/lgb_m3_32-50-0/lgb_m3_32-50-0.ftr')
    else:
        df_pred = loader.load_df('../../output/m3/lgb_m3_32-50-0/lgb_m3_32-50-0_cv.ftr')
    df_pred = df_pred[ID_NAMES + ['target']]
    
    df_pred = df_pred.sort_values(by=['target'], ascending=False)
    df_pred['pred_rank'] = df_pred.groupby(['description_id']).cumcount().values
    df_pred = df_pred.sort_values(by=['description_id', 'target'])
    print (df_pred.shape)
    print (df_pred.head(10))

    pred_top1 = df_pred[df_pred['pred_rank'] == 0] \
            .drop_duplicates(subset='description_id', keep='first')
    pred_top1 = pred_top1[['description_id', 'target']]
    pred_top1.columns = ['description_id', 'top1_pred']

    pred_top2 = df_pred[df_pred['pred_rank'] < 2]
    pred_top2['top2_pred_avg'] = pred_top2.groupby('description_id')['target'].transform('mean')
    pred_top2['top2_pred_std'] = pred_top2.groupby('description_id')['target'].transform('std')
    pred_top2 = pred_top2[['description_id', 'top2_pred_avg', \
            'top2_pred_std']].drop_duplicates(subset=['description_id'])

    pred_top3 = df_pred[df_pred['pred_rank'] < 3]
    pred_top3['top3_pred_avg'] = pred_top3.groupby('description_id')['target'].transform('mean')
    pred_top3['top3_pred_std'] = pred_top3.groupby('description_id')['target'].transform('std')
    pred_top3 = pred_top3[['description_id', 'top3_pred_avg', \
            'top3_pred_std']].drop_duplicates(subset=['description_id'])

    pred_top5 = df_pred[df_pred['pred_rank'] < 5]
    pred_top5['top5_pred_avg'] = pred_top5.groupby('description_id')['target'].transform('mean')
    pred_top5['top5_pred_std'] = pred_top5.groupby('description_id')['target'].transform('std')
    pred_top5 = pred_top5[['description_id', 'top5_pred_avg', \
            'top5_pred_std']].drop_duplicates(subset=['description_id'])

    df_pred.rename(columns={'target': 'pred'}, inplace=True)
    df = df.merge(df_pred, on=ID_NAMES, how='left')
    df = df.merge(pred_top1, on=['description_id'], how='left')
    df = df.merge(pred_top2, on=['description_id'], how='left')
    df = df.merge(pred_top3, on=['description_id'], how='left')
    df = df.merge(pred_top5, on=['description_id'], how='left')
    
    df['pred_sub_top1'] = df['pred'] - df['top1_pred']
    df['pred_sub_top2_avg'] = df['pred'] - df['top2_pred_avg']
    df['pred_sub_top3_avg'] = df['pred'] - df['top3_pred_avg']
    df['pred_sub_top5_avg'] = df['pred'] - df['top5_pred_avg']

    del_cols = ['paper_id', 'pred', 'pred_rank']
    df.drop(del_cols, axis=1, inplace=True)
    df_feat = df.drop_duplicates(subset=['description_id'])

    print ('df_feat info')
    print (df_feat.shape)
    print (df_feat.head())
    print (df_feat.columns.tolist())

    return df_feat

def output_fea(tr, te):
    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_fea_out_path)
    loader.save_df(te, te_fea_out_path)

def gen_fea():
    tr = loader.load_df('../../feat/tr_s0_32-50.ftr')
    te = loader.load_df('../../feat/te_s0_32-50.ftr')
        
    tr_feat = feat_extract(tr[ID_NAMES])
    te_feat = feat_extract(te[ID_NAMES], is_te=True)
    
    tr = tr[ID_NAMES].merge(tr_feat, on=['description_id'], how='left')
    te = te[ID_NAMES].merge(te_feat, on=['description_id'], how='left')
    
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
    base_tr_path = root_path + 'tr_s0_32-50.ftr'
    base_te_path = root_path + 'te_s0_32-50.ftr'

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


