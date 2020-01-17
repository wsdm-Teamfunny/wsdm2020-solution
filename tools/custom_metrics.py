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
from math import sqrt

# 评价指标
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def _calc_auc(labels, ypreds):
    return roc_auc_score(labels, ypreds)

def _calc_logloss(labels, ypreds):
    return log_loss(labels, ypreds)

def _calc_mae(labels, ypreds):
    return mean_absolute_error(labels, ypreds)

def _calc_rmse(labels, ypreds):
    return sqrt(mean_squared_error(labels, ypreds))

# kappa

# multi-logloss

def _calc_metric(labels, ypreds, metric_name='auc'):
    if metric_name == 'auc':
        return _calc_auc(labels, ypreds)
    elif metric_name == 'logloss':
        return _calc_logloss(labels, ypreds)
    elif metric_name == 'mae':
        return _calc_mae(labels, ypreds)
    elif metric_name == 'rmse':
        return _calc_rmse(labels, ypreds)
        
def calc_metrics(labels, ypreds, metric_names=['auc']):
    eval_lis = []
    for metric_name in metric_names:
        eval_val = _calc_metric(labels, ypreds, metric_name=metric_name)
        eval_val = round(eval_val, 5)
        eval_lis.append(eval_val)
    return eval_lis



