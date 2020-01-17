#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# 模型相关
import lightgbm as lgb
from basic_learner import BaseLearner

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

# 设置模型通用参数
EVAL_ROUND = 100
PRINT_TRAIN_METRICS = False


class lgbLearner(BaseLearner):

    def __init__(self, train_data, test_data, \
            fea_names, id_names, target_name, \
            params, num_trees, fold_num, out_name, \
            cv_name='cv', metric_names=['auc'], model_postfix=''):
        super(lgbLearner, self).__init__(train_data, test_data, fea_names, \
                id_names, target_name, params, fold_num, \
                out_name, metric_names, model_postfix)
        self.num_trees = num_trees
        self.cv_name = cv_name

        self.eval_round = EVAL_ROUND
        self.print_train_metrics = PRINT_TRAIN_METRICS

    def extract_train_data(self, data, predicted_fold_index):

        Xtrain = data[data[self.cv_name] != predicted_fold_index]
        Xvalid = data[data[self.cv_name] == predicted_fold_index]

        dtrain = lgb.Dataset(Xtrain[self.fea_names].values, \
                        Xtrain[self.target_name])
        dvalid = lgb.Dataset(Xvalid[self.fea_names].values, \
                        Xvalid[self.target_name])

        print ('train, valid', Xtrain.shape, Xvalid.shape)
        return dtrain, dvalid, Xvalid

    def train(self, data, predicted_fold_index, model_dump_path=None):
        if model_dump_path == None:
            model_dump_path = self.get_model_path(predicted_fold_index)

        dtrain, dvalid, Xvalid = self.extract_train_data(self.train_data,
                predicted_fold_index)

        if self.print_train_metrics:
            valid_sets = [dtrain, dvalid] \
                    if predicted_fold_index != 0 else [dtrain]
            valid_names = ['train', 'valid'] \
                    if predicted_fold_index != 0 else ['train']
        else:
            valid_sets = [dvalid] if predicted_fold_index != 0 else [dtrain]
            valid_names = ['valid'] if predicted_fold_index != 0 else ['train']

        params = self.params

        bst = lgb.train(params, dtrain, self.num_trees,
                valid_sets=valid_sets,
                valid_names=valid_names,
                verbose_eval=self.eval_round)
        bst.save_model(model_dump_path)

    def predict(self, data, predicted_fold_index, \
                model_load_path=None):
        if model_load_path is None:
            model_load_path = self.get_model_path(predicted_fold_index)

        bst = lgb.Booster(model_file=model_load_path)
        ypreds = bst.predict(data[self.fea_names], num_iteration=self.num_trees)

        if predicted_fold_index != 0:
            # output fea importance
            df = pd.DataFrame(self.fea_names, columns=['feature'])
            df['importance'] = list(bst.feature_importance('gain'))
            df['precent'] = np.round(df.importance * 100 / sum(df.importance), 2)
            df['precent'] = df.precent.apply(lambda x : str(x) + '%')

            df = df.sort_values(by='importance', ascending=False)
            imp_path = 'imp'
            if self.model_postfix != '':
                imp_path = 'imp-{}'.format(self.model_postfix)
            df.to_csv(self.root_path + imp_path, sep='\t')
        return ypreds

