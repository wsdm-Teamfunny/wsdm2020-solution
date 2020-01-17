#!/usr/bin/env python
#coding=utf-8

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# 基础模块
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

# 数据处理
import numpy as np
import pandas as pd

# 自定义工具包
sys.path.append('../../tools/')
import loader
import pandas_util

# 开源工具包
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# 设置随机种子
SEED = 2020

def topk_lines(df, k):
    df.loc[:, 'rank'] = df.groupby(['description_id']).cumcount().values
    df = df[df['rank'] < k]
    df.drop(['rank'], axis=1, inplace=True)
    return df

def process(in_path, k):
    ID_NAMES = ['description_id', 'paper_id']

    df = loader.load_df(in_path)
    df = topk_lines(df, k)
    df['sim_score'] = df['sim_score'].astype('float')
    df.rename(columns={'sim_score': 'corp_sim_score'}, inplace=True)
    return df


if __name__ == "__main__":

    ts = time.time()
    tr_path = '../../feat/tr_tfidf_30.ftr'
    te_path = '../../feat/te_tfidf_30.ftr'

    cv = loader.load_df('../../input/cv_ids_0109.csv')[['description_id', 'cv']]

    tr = process(tr_path, k=50)
    tr = tr.merge(cv, on=['description_id'], how='left')

    te = process(te_path, k=50)
    te['cv'] = 0

    loader.save_df(tr, '../../feat/tr_samples_30-50.ftr')
    loader.save_df(te, '../../feat/te_samples_30-50.ftr')
    print('all completed: {}, cost {}s'.format(datetime.now(), np.round(time.time() - ts, 2)))





