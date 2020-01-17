#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
import gc
import json
import time
import functools
from datetime import datetime

# 数据处理
import numpy as np
import pandas as pd
from math import sqrt
from collections import Counter

# 自定义工具包
sys.path.append('../../../tools/')
import loader
from lgb_learner import lgbLearner

# 设置随机种子
SEED = 2020
np.random.seed (SEED)

FEA_NUM = sys.argv[1]
FEA_NUM = '32-50'

fold_num = 5
out_name  = 'lgb_m3_{}-0'.format(FEA_NUM)
root_path = '../../../output/m3/' + out_name + '/'

ID_NAMES = ['description_id', 'paper_id']
TARGET_NAME = 'target'

TASK_TYPE = 'te'
#TASK_TYPE = 'tr'
#TASK_TYPE = 'pe'

if not os.path.exists(root_path):
    os.mkdir(root_path)
    print ('create dir succ {}'.format(root_path))

def sum_score(x, y):
    return max(x, 0) + max(y, 0)

def get_feas(data):

    cols = data.columns.tolist()
    del_cols = ID_NAMES + ['target', 'cv'] 
    sub_cols = ['year']
    for col in data.columns:
        for sub_col in sub_cols:
            #if sub_col in col and col != 'year':
            if sub_col in col:
                del_cols.append(col)

    cols = [val for val in cols if val not in del_cols]
    print ('del_cols', del_cols)
    return cols

def lgb_train(train_data, test_data, fea_col_names, seed=SEED, cv_index=0):
    params = {
        "objective":        "binary",
        "boosting_type":    "gbdt",
        #"metric":           ['binary_logloss'],
        "metric":           ['auc'],
        "boost_from_average": False,
        "learning_rate":    0.03,
        "num_leaves":       32,
        "max_depth":        -1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq":     2,
        "lambda_l1":        0,
        "lambda_l2":        0,
        "seed":             seed,
        'min_child_weight':  0.005,
        'min_data_in_leaf':  50,
        'max_bin':           255,
        "num_threads":       16,
        "verbose":          -1,
        "early_stopping_round": 50
    }
    params['learning_rate'] = 0.03
    num_trees = 2000
    print ('training params:', num_trees, params)

    lgb_learner = lgbLearner(train_data, test_data, \
            fea_col_names, ID_NAMES, TARGET_NAME, \
            params, num_trees, fold_num, out_name, \
            metric_names=['auc', 'logloss'], \
            model_postfix='')
    predicted_folds = [1,2,3,4,5]

    if TASK_TYPE == 'te':
        lgb_learner.multi_fold_train(lgb_learner.train_data, \
                predicted_folds=predicted_folds, need_predict_test=True)
    elif TASK_TYPE == 'tr':
        lgb_learner.multi_fold_train(lgb_learner.train_data, \
                predicted_folds=predicted_folds, need_predict_test=False)
    elif TASK_TYPE == 'pe':
        lgb_learner.multi_fold_predict(lgb_learner.train_data, \
                predicted_folds=predicted_folds, need_predict_test=False)

if __name__ == '__main__':

    ##################  params ####################
    print("Load the training, test and store data using pandas")
    ts = time.time()
    root_path = '../../../feat/'
    postfix = 's0_{}'.format(FEA_NUM)
    file_type = 'ftr'

    train_path = root_path + 'tr_{}.{}'.format(postfix, file_type)
    test_path  = root_path + 'te_{}.{}'.format('s0_4', file_type)
    if TASK_TYPE in ['te', 'pe']:
        test_path  = root_path + 'te_{}.{}'.format(postfix, file_type)

    print ('tr path', train_path)
    print ('te path', test_path)
    train_data = loader.load_df(train_path)
    test_data = loader.load_df(test_path)

    paper = loader.load_df('../../../input/candidate_paper_for_wsdm2020.ftr')
    tr = loader.load_df('../../../input/tr_input_final.ftr')
    tr = tr.merge(paper[['paper_id', 'journal', 'year']], on=['paper_id'], how='left')
    desc_list = tr[tr['journal'] != 'no-content'][~pd.isnull(tr['year'])]['description_id'].tolist()
    #train_data = train_data[train_data['description_id'].isin(desc_list)]
    
    print (train_data.columns)
    print (train_data.shape, test_data.shape)

    fea_col_names = get_feas(train_data)
    print (len(fea_col_names), fea_col_names)

    required_cols = ID_NAMES + ['cv', 'target']
    drop_cols = [col for col in train_data.columns \
            if col not in fea_col_names and col not in required_cols]

    train_data = train_data.drop(drop_cols, axis=1)
    test_data = test_data.drop([col for col in drop_cols if col in test_data.columns], axis=1)

    lgb_train(train_data, test_data, fea_col_names)
    print('all completed: %s, cost %s' % (datetime.now(), time.time() - ts))




