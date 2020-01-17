import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import matchzoo as mz
from model import ESIMplus

from gensim.models import KeyedVectors
from utils import MAP, build_matrix, topk_lines, predict

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('max_colwidth',400)


num_neg = 10
fit_preprocessor = True
losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
feature = [
    'quer_key_tfidf_corp_cos_dis',
    'quer_key_tfidf_corp_eucl_dis', 
    'quer_key_corp_bm25_score', 
    'corp_sim_score', 
    'quer_all_tfidf_corp_eucl_dis', 
    'quer_all_corp_bm25_score',
    'quer_key_tfidf_titl_manh_dis', 
    'quer_all_titl_bm25_score', 
    'quer_all_tfidf_corp_cos_dis',
    'jaccard_coef_of_unigram_between_corp_quer_key',
    'ratio_of_unique_corp_unigram', 
    'jaccard_coef_of_unigram_between_corp_quer_all', 
    'jaccard_coef_of_unigram_between_titl_quer_key',
    'quer_key_tfidf_titl_cos_dis', 
    'jaccard_coef_of_unigram_between_abst_quer_key', 
    'quer_key_abst_bm25_score', 
    'quer_all_tfidf_titl_cos_dis', 
    'quer_key_tfidf_titl_eucl_dis', 
    'count_of_quer_key_unigram', 
    'quer_all_tfidf_titl_eucl_dis', 
    'ratio_of_unique_quer_all_unigram', 
    'quer_key_tfidf_abst_cos_dis', 
    'count_of_unique_corp_unigram', 
    'ratio_of_unique_abst_unigram', 
    'normalized_pos_of_corp_unigram_in_quer_all_max', 
    'quer_all_abst_bm25_score', 
    'normalized_pos_of_titl_unigram_in_quer_all_std',
    'quer_all_tfidf_titl_manh_dis', 
    'jaccard_coef_of_unigram_between_abst_quer_all', 
    'dice_dist_of_unigram_between_corp_quer_key']

task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]
print("task is", task)
print("`task` initialized with metrics", task.metrics)

if fit_preprocessor:
    
    preprocessor = mz.models.ESIM.get_default_preprocessor(
        truncated_mode='pre',
        truncated_length_left=64,
        truncated_length_right=256,
        filter_mode='df',
        filter_low_freq=2)
    
    preprocessor = preprocessor.fit(all_data_raw)
    preprocessor.save("preprocessor.prep")
else:
    preprocessor = mz.load_preprocessor("preprocessor.prep")


candidate_dic = pd.read_feather('data/candidate_dic.ftr')

train_recall = pd.read_feather('data/train_recall.ftr')
train_description = pd.read_feather('data/train_description.ftr')
train_recall = pd.merge(train_recall, train_description, how='left', on='id_left')
train_recall = pd.merge(train_recall, candidate_dic, how='left', on='id_right')
train_recall = train_recall.drop_duplicates().reset_index(drop=True)
del train_description
gc.collect()


test_recall = pd.read_feather('data/test_recall.ftr')
test_description = pd.read_feather('data/test_description.ftr')
test_recall = pd.merge(test_recall, test_description, how='left', on='id_left')
test_recall = pd.merge(test_recall, candidate_dic, how='left', on='id_right')
del test_description, candidate_dic
gc.collect()

all_data_df = train_recall.copy()
all_data_df.id_left = all_data_df.id_left+'_tr'
all_data_df = pd.concat([all_data_df, test_recall]).reset_index(drop=True)
norm_df = all_data_df[feature].quantile(q=0.99)

del all_data_df, train_recall, test_recall
gc.collect()

train_recall[feature] = train_recall[feature]/norm_df
train_recall['feature'] = list(train_recall[feature].values)
train_recall = train_recall[['id_left', 'text_left', 'id_right', 'text_right', 'label', 'feature']]
cv_ids = pd.read_csv("../../input/cv_ids_0109.csv")
train_recall = train_recall.merge(
    cv_ids.rename(columns={'description_id': 'id_left'}),
    how='left', 
    on='id_left').fillna(5.0)


for i in range(1,6):
    print("="*20, i, "="*20)
    train_df = train_recall[train_recall.cv!=i][
        ['id_left', 'text_left', 'id_right', 'text_right', 'label', 'feature']].reset_index(drop=True)
    val_df = train_recall[train_recall.cv==i][
        ['id_left', 'text_left', 'id_right', 'text_right', 'label', 'feature']].reset_index(drop=True)

    train_raw = mz.pack(train_df, task)
    val_raw = mz.pack(val_df, task)
    
    train_processed = preprocessor.transform(train_raw)
    val_processed = preprocessor.transform(val_raw)
    
    train_processed.save("5fold/train_processed_{}.dp".format(i))
    val_processed.save("5fold/val_processed_{}.dp".format(i))
    
    
test_recall[feature] = test_recall[feature]/norm_df
test_recall['feature'] = list(test_recall[feature].values)
test_recall = test_recall[['id_left', 'text_left', 'id_right', 'text_right', 'feature']]

test_raw = mz.pack(test_recall, task)
test_processed = preprocessor.transform(test_raw)
# test_processed.save("test_processed.dp")
test_processed.save("final_test_processed.dp")


from gensim.models import KeyedVectors
w2v_path = "data/glove.w2v"
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = build_matrix(term_index, w2v_model)
del w2v_model, term_index
gc.collect()
np.save("data/embedding_matrix.npy", embedding_matrix) 
    

