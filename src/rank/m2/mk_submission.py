import numpy as np
import pandas as pd
from tqdm import tqdm

test_recall = pd.read_feather('../../feat/te_s0_32-50.ftr')[['description_id', 'paper_id', 'corp_sim_score']]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='ESIMplus_001')
args = parser.parse_args()
model_id = args.model_id

if '_pointwise' in model_id:
    fold = 1
    test_df = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
        columns={'target':'target_1'})

    for fold in tqdm(range(2,6)):
        test_df_cv = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
            columns={'target':'target_{}'.format(fold)})
        test_df = test_df.merge(test_df_cv)

    score_cols = ['target_1', 'target_2', 'target_3', 'target_4', 'target_5']
    test_df['target'] = test_df[score_cols].mean(axis=1)
    print(test_df[score_cols+['target']].corr(method='spearman'))
else:    
    fold = 1
    test_df = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
        columns={'score':'score_1'})

    for fold in tqdm(range(2,6)):
        test_df_cv = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
            columns={'score':'score_{}'.format(fold)})
        test_df = test_df.merge(test_df_cv)

    score_cols = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5']
    test_df['score'] = test_df[score_cols].mean(axis=1)
    print(test_df[score_cols+['score']].corr(method='spearman'))


if 'target' not in test_df.columns:
    test_df['target'] = test_df['score'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
    
test_df = test_recall.merge(
    test_df[['description_id', 'paper_id', 'target']], how='left', on=['description_id', 'paper_id'])
test_df[['description_id', 'paper_id', 'target']].to_csv("final_result_m2_{}_5cv.csv".format(model_id), index=False)

result = test_df.sort_values(by=['description_id', 'target', 'corp_sim_score'], na_position='first').groupby(
    'description_id').tail(3)

description_id_list = []
paper_id_list_1 = []
paper_id_list_2 = []
paper_id_list_3 = []
for description_id, df_tmp in tqdm(result.groupby('description_id')):
    description_id_list.append(description_id)
    paper_id_list_1.append(df_tmp.iloc[2,1])
    paper_id_list_2.append(df_tmp.iloc[1,1])
    paper_id_list_3.append(df_tmp.iloc[0,1])
    
sub = pd.DataFrame(data={'description_id':description_id_list, 
                         'paper_id_1': paper_id_list_1, 
                         'paper_id_2': paper_id_list_2, 
                         'paper_id_3': paper_id_list_3})
sub.to_csv("final_{}_sub_5cv.csv".format(model_id), header=False, index=False)
print("final_{}_sub_5cv.csv".format(model_id))











