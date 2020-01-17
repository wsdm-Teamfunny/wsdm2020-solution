#!/usr/bin/env python
#coding=utf-8

# bm25 recall

# 基础模块
import os
import gc
import sys
import time
import functools
from tqdm import tqdm
from six import iteritems
from datetime import datetime

# 数据处理
import re
import math
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool

# 自定义工具包
sys.path.append('../../tools/')
import loader
import pandas_util
import custom_bm25 as bm25

# 开源工具包
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# 设置随机种子
SEED = 2020
PROCESS_NUM, PARTITION_NUM = 18, 18

input_root_path  = '../../input/'
output_root_path = '../../feat/'

postfix = '30'
file_type = 'ftr'

train_out_path = output_root_path + 'tr_tfidf_{}.{}'.format(postfix, file_type)
test_out_path = output_root_path + 'te_tfidf_{}.{}'.format(postfix, file_type)

def topk_sim_samples(desc, desc_ids, paper_ids, bm25_model, k=10):
    desc_id2papers = {}
    for desc_i in tqdm(range(len(desc))):
        query_vec, query_desc_id = desc[desc_i], desc_ids[desc_i]
        sims = bm25_model.get_scores(query_vec)
        sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
        sim_papers = [paper_ids[val[0]] for val in sort_sims[:k]]
        sim_scores = [str(val[1]) for val in sort_sims[:k]]
        desc_id2papers[query_desc_id] = ['|'.join(sim_papers), '|'.join(sim_scores)]
    sim_df = pd.DataFrame.from_dict(desc_id2papers, orient='index', columns=['paper_id', 'sim_score'])
    sim_df = sim_df.reset_index().rename(columns={'index':'description_id'})
    return sim_df

def partition(queries, num):
    queries_partitions, step = [], int(np.ceil(len(queries)/num))
    for i in range(0, len(queries), step):
        queries_partitions.append(queries[i:i+step])
    return queries_partitions

def single_process_search(params=None):
    (query_vecs, desc_ids, paper_ids, bm25_model, k, i) = params
    print (i, 'start', datetime.now())
    gc.collect()
    sim_df = topk_sim_samples(query_vecs, desc_ids, paper_ids, bm25_model, k)
    print (i, 'completed', datetime.now())
    return sim_df

def multi_process_search(query_vecs, desc_ids, paper_ids, bm25_model, k):
    pool = Pool(PROCESS_NUM)
    queries_parts = partition(query_vecs, PARTITION_NUM)
    desc_ids_parts = partition(desc_ids, PARTITION_NUM)
    print ('{} processes init and partition to {} parts' \
           .format(PROCESS_NUM, PARTITION_NUM))

    param_list = [(queries_parts[i], desc_ids_parts[i], \
        paper_ids, bm25_model, k, i) for i in range(PARTITION_NUM)]
    sim_dfs = pool.map(single_process_search, param_list)
    sim_df = pd.concat(sim_dfs, axis=0)
    return sim_df

def gen_samples(df, desc, desc_ids, corpus_list, paper_ids_list, k):
    df_samples_list = []
    for i, corpus in enumerate(corpus_list):
        bm25_model = bm25.BM25(corpus[0])        
        cur_df_sample = multi_process_search(desc, desc_ids, \
                paper_ids_list[i], bm25_model, k)
        cur_df_sample_out = pandas_util.explode(cur_df_sample, ['paper_id', 'sim_score'])
        cur_df_sample_out['type'] = corpus[1] # recall_name
        df_samples_list.append(cur_df_sample_out)
    df_samples = pd.concat(df_samples_list, axis=0)
    df_samples.drop_duplicates(subset=['description_id', 'paper_id'], inplace=True)
    df_samples['target'] = 0
    return df_samples

if __name__ == "__main__":

    ts = time.time()
    tqdm.pandas()
    print('start time: %s' % datetime.now())
    # load data
    df = loader.load_df(input_root_path + 'paper_input_final.ftr')
    df = df[~pd.isnull(df['paper_id'])]
                               
    # gen tfidf vecs
    dictionary = pickle.load(open('../../feat/corpus.dict', 'rb'))
    print ('dic len', len(dictionary))

    df['corp'] = df['abst'] + ' ' + df['titl'] + ' ' + df['keywords'].fillna('').replace(';', ' ')
    df_corp, corp_paper_ids = [dictionary.doc2bow(line.split(' ')) for line in df['corp'].tolist()], \
            df['paper_id'].tolist()
       
    # gen topk sim samples
    paper_ids_list = [corp_paper_ids]
    corpus_list = [(df_corp, 'corp_bm25')]
    out_cols = ['description_id', 'paper_id', 'sim_score', 'target', 'type']

    if sys.argv[1] in ['tr']:
        # for tr ins
        tr = loader.load_df(input_root_path + 'tr_input_final.ftr')
        tr = tr[~pd.isnull(tr['description_id'])]
        
#         tr = tr.head(1000)        
        tr_desc, tr_desc_ids = [dictionary.doc2bow(line.split(' ')) for line in tr['quer_all'].tolist()], \
                tr['description_id'].tolist()
        print ('gen tf completed, cost {}s'.format(np.round(time.time() - ts, 2)))   
        
        tr_samples = gen_samples(tr, tr_desc, tr_desc_ids, \
                corpus_list, paper_ids_list, k=50)
        tr_samples = tr.rename(columns={'paper_id': 'target_paper_id'}) \
                .merge(tr_samples, on='description_id', how='left')
        tr_samples.loc[tr_samples['target_paper_id'] == tr_samples['paper_id'], 'target'] = 1
        loader.save_df(tr_samples[out_cols], train_out_path)
        print ('recall succ {} from {}'.format(tr_samples['target'].sum(), tr.shape[0]))
        print (tr.shape, tr_samples.shape)

    if sys.argv[1] in ['te']:
        # for te ins
        te = loader.load_df(input_root_path + 'te_input_final.ftr')
        te = te[~pd.isnull(te['description_id'])]
        
#         te = te.head(1000)
        te_desc, te_desc_ids = [dictionary.doc2bow(line.split(' ')) for line in te['quer_all'].tolist()], \
                te['description_id'].tolist()
        print ('gen tf completed, cost {}s'.format(np.round(time.time() - ts, 2)))
                
        te_samples = gen_samples(te, te_desc, te_desc_ids, \
                corpus_list, paper_ids_list, k=50)
        te_samples = te.merge(te_samples, on='description_id', how='left')
        loader.save_df(te_samples[out_cols], test_out_path)
        print (te.shape, te_samples.shape)
        
    print('all completed: {}, cost {}s'.format(datetime.now(), np.round(time.time() - ts, 2)))



