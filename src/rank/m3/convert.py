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

# 自定义工具包
sys.path.append('../../../tools/')
import loader

# 设置随机种子
SEED = 2020
np.random.seed (SEED)

def val_convert(df_path, pred_path, out_path):
    tr_data = loader.load_df(df_path)
    df_pred = loader.load_df(pred_path)

    sort_df_pred = df_pred.sort_values(['description_id', 'target'], ascending=False)
    df_pred = df_pred[['description_id']].drop_duplicates() \
            .merge(sort_df_pred, on=['description_id'], how='left')
    df_pred['rank'] = df_pred.groupby('description_id').cumcount().values
    df_pred = df_pred[df_pred['rank'] < 3]
    df_pred = df_pred.groupby(['description_id'])['paper_id'] \
            .apply(lambda s : ','.join((s))).reset_index()

    tr_data = tr_data[['description_id', 'paper_id']].rename(columns={'paper_id': 'target_id'})
    df_pred = df_pred.merge(tr_data, on=['description_id'], how='left')
    loader.save_df(df_pred, out_path)

def output(df, out_path):
    fo = open(out_path, 'w')
    for i in range(df.shape[0]):
        desc_id = df.iloc[i]['description_id']
        paper_ids = df.iloc[i]['paper_id']
        print (desc_id + ',' + paper_ids, file=fo)
    fo.close()

def sub_convert(df_path, pred_path, out_path1, out_path2):
    te_data = loader.load_df(df_path)
    df_pred = loader.load_df(pred_path)

    sort_df_pred = df_pred.sort_values(['description_id', 'target'], ascending=False)
    df_pred = df_pred[['description_id']].drop_duplicates() \
            .merge(sort_df_pred, on=['description_id'], how='left')
    df_pred['rank'] = df_pred.groupby('description_id').cumcount().values
    df_pred = df_pred[df_pred['rank'] < 3]
    df_pred = df_pred.groupby(['description_id'])['paper_id'] \
            .apply(lambda s : ','.join((s))).reset_index()

    df_pred = te_data[['description_id']].merge(df_pred, on=['description_id'], how='left')
    loader.save_df(df_pred, out_path1)
    #output(df_pred, out_path2)

if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../../../feat/'
    base_tr_path = '../../../input/train_release.csv'
    base_te_path = '../../../input/test.csv'

    sub_file_path = sys.argv[1]
    sub_name = sys.argv[2]

    val_path = '{}/{}_cv.ftr'.format(sub_file_path, sub_name)
    val_out_path = '{}/r_{}_cv.csv'.format(sub_file_path, sub_name)
    val_convert(base_tr_path, val_path, val_out_path)

    sub_path = '{}/{}.ftr'.format(sub_file_path, sub_name)
    sub_out_pathA = '{}/r_{}.csv'.format(sub_file_path, sub_name)
    sub_out_pathB = '{}/s_{}.csv'.format(sub_file_path, sub_name)
    sub_out_pathA2 = '{}/r2_{}.csv'.format(sub_file_path, sub_name)
    sub_out_pathB2 = '{}/s2_{}.csv'.format(sub_file_path, sub_name)
    sub_convert(base_te_path, sub_path, sub_out_pathA, sub_out_pathB)

    print('all completed: %s' % datetime.now())


