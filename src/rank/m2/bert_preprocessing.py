import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import matchzoo as mz
from matchzoo.preprocessors.units.truncated_length import TruncatedLength
from utils import MAP, build_matrix, topk_lines, predict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessing_type', type=str, default='fine')
parser.add_argument('--left_truncated_length', type=int, default=64)
parser.add_argument('--query_type', type=str, default='query_key')
args = parser.parse_args()

preprocessing_type = args.preprocessing_type
left_truncated_length = args.left_truncated_length
dp_type = args.query_type

num_neg = 7
losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]

preprocessor = mz.models.Bert.get_default_preprocessor(mode='bert-base-uncased')


if preprocessing_type == 'fine':
    candidate_dic = pd.read_feather('data/candidate_dic.ftr')
    train_description = pd.read_feather('data/train_description_{}.ftr'.format(dp_type))
else:
    candidate_dic = pd.read_csv('../../../input/candidate_paper_for_wsdm2020.csv')
    candidate_dic.loc[candidate_dic['keywords'].isna(),'keywords'] = ''
    candidate_dic.loc[candidate_dic['title'].isna(),'title'] = ''
    candidate_dic.loc[candidate_dic['abstract'].isna(),'abstract'] = ''
    candidate_dic['text_right'] = candidate_dic['abstract'].str.cat(
        candidate_dic['keywords'], sep=' ').str.cat(
        candidate_dic['title'], sep=' ')
    candidate_dic = candidate_dic.rename(columns={'paper_id': 'id_right'})[['id_right', 'text_right']]

    train_description = pd.read_csv('../../../input/train_release.csv')
    train_description = train_description.rename(
        columns={'description_id': 'id_left', 
                 'description_text': 'text_left'})[['id_left', 'text_left']]
    dp_type = 'query_all_nopreprocessing'

train_recall = pd.read_feather('data/train_recall.ftr')[['id_left', 'id_right', 'label', 'cv']]
train_recall = pd.merge(train_recall, train_description, how='left', on='id_left')
train_recall = pd.merge(train_recall, candidate_dic, how='left', on='id_right')
train_recall = train_recall.drop_duplicates().reset_index(drop=True)
train_recall = train_recall[['id_left', 'text_left', 'id_right', 'text_right', 'label', 'cv']]
del train_description
gc.collect()



for i in range(1,6):
    print("="*20, i, "="*20)
    train_df = train_recall[train_recall.cv!=i][
        ['id_left', 'text_left', 'id_right', 'text_right', 'label']].reset_index(drop=True)
    val_df = train_recall[train_recall.cv==i][
        ['id_left', 'text_left', 'id_right', 'text_right', 'label']].reset_index(drop=True)
    
    train_raw = mz.pack(train_df, task)
    train_processed = preprocessor.transform(train_raw)
    train_processed.apply_on_text(TruncatedLength(left_truncated_length, 'pre').transform, 
                                  mode='left', inplace=True, verbose=1)
    train_processed.apply_on_text(TruncatedLength(256, 'pre').transform, mode='right', inplace=True, verbose=1)
    train_processed.append_text_length(inplace=True, verbose=1)
    train_processed.save("bert_data/bert_train_processed_{}_{}.dp".format(dp_type, i))

    val_raw = mz.pack(val_df, task)
    val_processed = preprocessor.transform(val_raw)
    val_processed.apply_on_text(TruncatedLength(left_truncated_length, 'pre').transform,
                                mode='left', inplace=True, verbose=1)
    val_processed.apply_on_text(TruncatedLength(256, 'pre').transform, mode='right', inplace=True, verbose=1)
    val_processed.append_text_length(inplace=True, verbose=1)
    val_processed.save("bert_data/bert_val_processed_{}_{}.dp".format(dp_type, i))


if preprocessing_type == 'fine':
    test_description = pd.read_feather('data/test_description_quer_all.ftr')
else:
    test_description = pd.read_csv('../../input/test.csv')
    test_description = test_description.rename(
        columns={'description_id': 'id_left', 
                 'description_text': 'text_left'})[['id_left', 'text_left']]


test_recall = pd.read_feather('data/test_recall.ftr')[['id_left', 'id_right', 'label']]
test_recall = pd.merge(test_recall, test_description, how='left', on='id_left')
test_recall = pd.merge(test_recall, candidate_dic, how='left', on='id_right')
del test_description, candidate_dic
gc.collect()

test_raw = mz.pack(test_recall, task)
test_processed = preprocessor.transform(test_raw)
test_processed.apply_on_text(TruncatedLength(left_truncated_length, 'pre').transform, 
                             mode='left', inplace=True, verbose=1)
test_processed.apply_on_text(TruncatedLength(256, 'pre').transform, mode='right', inplace=True, verbose=1)
test_processed.append_text_length(inplace=True, verbose=1)
test_processed.save("bert_data/bert_test_processed_{}.dp".format(dp_type))
    
    
    
    