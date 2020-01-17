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
import loader
import custom_metrics

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

class BaseLearner(object):

    def __init__(self, train_data, test_data, 
            fea_names, id_names, target_name, \
            params, fold_num, out_name, metric_names=['auc'], \
            model_postfix=''):
        # 深度拷贝原始数据，防止外部主函数修改导致的数据异常
        self.train_data = train_data.copy(deep=True)
        self.test_data = test_data.copy(deep=True)

        # 基础数据信息
        self.fea_names = fea_names
        self.id_names = id_names
        self.target_name = target_name

        self.params = params
        self.fold_num = fold_num
        self.out_name = out_name
        self.root_path = '../../../output/m3/' + out_name + '/'
        self.metric_names = metric_names
        self.model_postfix = model_postfix

    # 获取模型存储路径
    def get_model_path(self, predicted_fold_index):
        model_path = self.root_path + 'model_' + str(predicted_fold_index)
        if self.model_postfix != '':
            model_path += '_' + self.model_postfix
        return model_path

    # 获取预测结果输出路径
    def get_preds_outpath(self, predicted_fold_index):
        out_path = self.root_path + self.out_name
        if self.model_postfix != '': 
            out_path += '_' + self.model_postfix
        if predicted_fold_index != 0:
            out_path += '_cv_' + str(predicted_fold_index)
        return out_path

    # 训练、验证集划分接口，需要被重载
    def extract_train_data(self, data, predicted_fold_index):
        pass 

    # 单 fold 训练接口，需要被重载
    def train(self, data, predicted_fold_index, model_dump_path=None):
        pass

    # 单 fold 预测接口，需要被重载
    def predict(self, data, predicted_fold_index, model_load_path=None):
        pass

    # 多 fold 训练
    def multi_fold_train(self, data, predicted_folds=[1,2,3,4,5], \
            need_predict_test=False):
        print ("multi_fold train start {}".format(datetime.now()))
        ts = time.time()
        for fold_index in predicted_folds:
            print ('training fold {}'.format(fold_index))
            self.train(data, fold_index)
            print ('fold {} completed, cost {}s'.format( \
                    fold_index, time.time() - ts))
        self.multi_fold_predict(data, predicted_folds, need_predict_test)

    # 多 fold 预测
    def multi_fold_predict(self, data, predicted_folds, \
            need_predict_test=False):
        print ("multi_fold predict start {}".format(datetime.now()))

        multi_fold_eval_lis = []
        
        for fold_index in predicted_folds:
            dtrain, dvalid, Xvalid = self.extract_train_data( \
                    self.train_data, fold_index)

            ypreds = self.predict(Xvalid, fold_index)
            labels = Xvalid[self.target_name]

            eval_lis = custom_metrics.calc_metrics(labels, ypreds, \
                    self.metric_names)

            multi_fold_eval_lis.append(eval_lis)
            print ('{} eval: {}'.format(fold_index, eval_lis))
            loader.out_preds(self.target_name, \
                    Xvalid[self.id_names], ypreds, \
                    '{}.csv'.format(self.get_preds_outpath(fold_index)), \
                    labels.tolist())

            if need_predict_test:
                print ('predict test data')
                ypreds = self.predict(self.test_data, 0,
                        model_load_path=self.get_model_path(fold_index))
                # output preds
                loader.out_preds(self.target_name, \
                        self.test_data[self.id_names], ypreds, \
                        '{}_{}.csv'.format(self.get_preds_outpath(0), fold_index))

        multi_fold_eval_avgs = []
        for i in range(len(self.metric_names)):
            eval_avg = np.array([val[i] for val in multi_fold_eval_lis]).mean()
            eval_avg = round(eval_avg, 5)
            multi_fold_eval_avgs.append(eval_avg)
        print ('multi fold eval mean: ', multi_fold_eval_avgs)

        return multi_fold_eval_avgs


