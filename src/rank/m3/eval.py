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

# 开源工具包
import ml_metrics as metrics

# 设置随机种子
SEED = 2020
np.random.seed (SEED)

def calc_map(df, k):
    df.rename(columns={'paper_id': 'paper_ids'}, inplace=True)
    df['paper_ids'] = df['paper_ids'].apply(lambda s: s.split(','))
    df['target_id'] = df['target_id'].apply(lambda s: [s])
    return metrics.mapk(df['target_id'].tolist(), df['paper_ids'].tolist(), k)

if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    in_path = sys.argv[1]
    df = loader.load_df(in_path)
    mapk = calc_map(df, k=3)
    print ('{} {}'.format(df.shape, round(mapk, 5)))
    print('all completed: %s' % datetime.now())

