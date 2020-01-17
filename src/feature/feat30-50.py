#!/usr/bin/env python
#coding=utf-8

# 根据召回结果生成基础特征（不包含词向量距离）

# 基础模块
import os
import gc
import sys
import time
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
from feat_utils import try_divide, dump_feat_name

# 开源工具包
import nltk
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# 设置随机种子
SEED = 2020

input_root_path  = '../../input/'
output_root_path = '../../feat/'

postfix = '30-50'
file_type = 'ftr'

tr_out_path = output_root_path + 'tr_s0_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_s0_{}.{}'.format(postfix, file_type)

ID_NAMES = ['description_id', 'paper_id']
PROCESS_NUM = 15

def preprocess_data(line):
    tokens = line.split(' ') if line else []
    return tokens

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target

def ngram_feat(df):    
    ## unigram
    print ("generate unigram")
    df["quer_key_unigram"] = list(df.progress_apply(lambda x: preprocess_data(x["quer_key"]), axis=1))
    df["quer_all_unigram"] = list(df.progress_apply(lambda x: preprocess_data(x["quer_all"]), axis=1))
    df["titl_unigram"] = list(df.progress_apply(lambda x: preprocess_data(x["titl"]), axis=1))
    df["abst_unigram"] = list(df.progress_apply(lambda x: preprocess_data(x["abst"]), axis=1))
    df["corp_unigram"] = list(df.progress_apply(lambda x: preprocess_data(x["corp"]), axis=1))
    return df

def count_feat(df):
    ################################
    ## word count and digit count ##
    ################################
    print ("generate word counting features")
    feat_names = ['quer_key', 'quer_all', 'titl', 'abst', 'corp']
    grams = ['unigram']
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            ## word count
            df["count_of_%s_%s"%(feat_name,gram)] = list(df.progress_apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
            df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.progress_apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
            df["ratio_of_unique_%s_%s"%(feat_name,gram)] = list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

        ## digit count
        df["count_of_digit_in_%s"%feat_name] = list(df.progress_apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
        df["ratio_of_digit_in_%s"%feat_name] = list(map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)]))    
    return df

def intersect_position_feat(df):
    ######################################
    ## intersect word position feat ##
    ######################################
    grams = ['unigram']
    print ("generate intersect word position features")
    for gram in grams:
        for obs_name in ['quer_key', 'quer_all']:
            for target_name in ['titl', 'abst', 'corp']:
                pos = list(df.progress_apply(lambda x: get_position_list(x[target_name+"_"+gram], \
                         obs=x[obs_name+"_"+gram]), axis=1))
                ## stats feat on pos
                df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = list(map(np.min, pos))
                df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = list(map(np.mean, pos))
                df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = list(map(np.median, pos))
                df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = list(map(np.max, pos))
                df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = list(map(np.std, pos))
                ## stats feat on normalized_pos
                df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_min" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_mean" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_median" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_max" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_std" % \
                        (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)]))
                
        for obs_name in ['titl', 'abst', 'corp']:
            for target_name in ['quer_key', 'quer_all']:
                pos = list(df.progress_apply(lambda x: get_position_list(x[target_name+"_"+gram], \
                         obs=x[obs_name+"_"+gram]), axis=1))
                ## stats feat on pos
                df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = list(map(np.min, pos))
                df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = list(map(np.mean, pos))
                df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = list(map(np.median, pos))
                df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = list(map(np.max, pos))
                df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = list(map(np.std, pos))
                ## stats feat on normalized_pos
                df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_min" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_mean" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_median" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_max" % \
                        (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)]))
                df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = \
                        list(map(try_divide, df["pos_of_%s_%s_in_%s_std" % \
                        (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)]))
    return df

#####################
## Distance metric ##
#####################
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = try_divide(intersect, union)
    return coef

def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d

#### pairwise distance
def pairwise_jaccard_coef(A, B):
    coef = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            coef[i,j] = JaccardCoef(A[i], B[j])
    return coef
    
def pairwise_dice_dist(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=float)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d[i,j] = DiceDist(A[i], B[j])
    return d

def pairwise_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = pairwise_jaccard_coef(A, B)
    elif dist == "dice_dist":
        d = pairwise_dice_dist(A, B)
    return d

#####################################
## Extract basic distance features ##
#####################################
def basic_distance_feat(df):
    ## jaccard coef/dice dist of n-gram
    print ("generate jaccard coef and dice dist for n-gram")
    dists = ['jaccard_coef', 'dice_dist']
    grams = ['unigram']
    for dist in dists:
        for gram in grams:
            for obs_name in ['quer_key', 'quer_all']:
                for target_name in ['titl', 'abst', 'corp']:
                    df["%s_of_%s_between_%s_%s"%(dist, gram, target_name, obs_name)] = \
                            list(df.apply(lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1))
    return df
                    
def single_process_feat(params=None):
    (df, i) = params
    print (i, 'start', datetime.now())
    df = ngram_feat(df)
    print(i, 'ngram completed', datetime.now())
    df = count_feat(df)
    print(i, 'count_feat completed', datetime.now())
    df = basic_distance_feat(df)
    print(i, 'basic_distance_feat completed', datetime.now())
    df = intersect_position_feat(df)
    print(i, 'intersect_position_feat completed', datetime.now())

    required_cols = ID_NAMES + ['cv']
    del_cols = [col for col in df.columns if df[col].dtype == 'O' and col not in required_cols]
    if i == 0:
        print (del_cols)
    df.drop(del_cols, axis=1, inplace=True)
    print (i, 'completed', datetime.now())
    return df

def partition(df, num):
    df_partitions, step = [], int(np.ceil(df.shape[0]/num))
    for i in range(0, df.shape[0], step):
        df_partitions.append(df.iloc[i:i+step])
    return df_partitions

def multi_process_feat(df):
    pool = Pool(PROCESS_NUM)
    df_parts = partition(df, PROCESS_NUM)

    print ('{} processes init and partition to {} parts' \
           .format(PROCESS_NUM, PROCESS_NUM))

    param_list = [(df_parts[i], i) for i in range(PROCESS_NUM)]
    dfs = pool.map(single_process_feat, param_list)
    df_out = pd.concat(dfs, axis=0)
    return df_out

def gen_samples(paper, tr_desc_path, tr_recall_path):
    tr_desc = loader.load_df(tr_desc_path)
    tr = loader.load_df(tr_recall_path)
    tr = tr.head(1000)
    
    tr = tr.merge(paper, on=['paper_id'], how='left')
    tr = tr.merge(tr_desc[['description_id', 'quer_key', 'quer_all']], on=['description_id'], how='left')

    print (tr.columns)
    print (tr.head())
    
    tr = multi_process_feat(tr)
    del_cols = [col for col in tr.columns if tr[col].dtype == 'O' and col not in ID_NAMES]
    print ('tr del cols', del_cols)
    return tr.drop(del_cols, axis=1)

if __name__ == "__main__":

    ts = time.time()
    tqdm.pandas()
    print('start time: %s' % datetime.now())    
    paper = loader.load_df('../../input/paper_input_final.ftr')
    paper['abst'] = paper['abst'].apply(lambda s: s.replace('no_content', ''))
    paper['corp'] = paper['abst'] + ' ' + paper['titl'] + ' ' + paper['keywords'].fillna('').replace(';', ' ')
    
    tr_desc_path = '../../input/tr_input_final.ftr'
    te_desc_path = '../../input/te_input_final.ftr'
   
    tr_recall_path = '../../feat/tr_samples_30-50.ftr'
    te_recall_path = '../../feat/te_samples_30-50.ftr'
    
    tr = gen_samples(paper, tr_desc_path, tr_recall_path)
    print (tr.columns)
    loader.save_df(tr, tr_out_path)
    
    te = gen_samples(paper, te_desc_path, te_recall_path)
    print (te.columns)
    loader.save_df(te, te_out_path)
    print('all completed: {}, cost {}s'.format(datetime.now(), np.round(time.time() - ts, 2)))




