import numpy as np
import pandas as pd
from tqdm import tqdm


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='ESIMplus_001')
args = parser.parse_args()
model_id = args.model_id


def map3_func(df, topk = 50, verbose=0):
    ids = df[df.label==1].description_id.values
    df_recalled = df[df.description_id.isin(ids)].reset_index(drop=True)
    df_recalled = df_recalled.sort_values(
        by=['description_id', 'label'], ascending=False).reset_index(drop=True)
    result = df_recalled.score.values.reshape([-1,topk])
    ranks = topk-result.argsort(axis=1).argsort(axis=1)
    map3_sum = sum(((1/ranks[:,0])*(ranks[:,0]<4)))
    if verbose>1:
        print("recall rate: "+str((df_recalled.shape[0]/topk)/(df.shape[0]/topk)))
        print("map@3 in recall: "+str(map3_sum/(df_recalled.shape[0]/topk)))
    if verbose>0:
        print("map@3 in all: "+str(map3_sum/(df.shape[0]/topk)))
    return map3_sum/(df.shape[0]/topk)


fold = 1
val_df = pd.read_csv("result/{}/{}_fold_{}_cv.csv".format(model_id, model_id, fold))
test_df = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
    columns={'score':'score_1'})

for fold in tqdm(range(2,6)):
    val_df_cv = pd.read_csv("result/{}/{}_fold_{}_cv.csv".format(model_id, model_id, fold))
    val_df = pd.concat([val_df, val_df_cv], ignore_index=True, sort=True)
    
    test_df_cv = pd.read_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold)).rename(
        columns={'score':'score_{}'.format(fold)})
    test_df = test_df.merge(test_df_cv)
    
val_df = val_df.merge(train_recall, how='left')
val_df = val_df[val_df.description_id!='6.45E+04'].reset_index(drop=True)
# assert val_df.description_id.nunique()==49945
map3_func(val_df)
val_df['target'] = val_df['score'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
val_df.to_csv("oof_m2_{}_5cv.csv".format(model_id), index=False)

score_cols = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5']
test_df['score'] = test_df[score_cols].mean(axis=1)
print(test_df[score_cols+['score']].corr(method='spearman'))

test_df['target'] = test_df['score'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
val_df['target'] = val_df['score'].apply(lambda x: np.exp(x)/(1+np.exp(x)))

test_df = test_recall.merge(
    test_df[['description_id', 'paper_id', 'score']], how='left', on=['description_id', 'paper_id'])
test_df['target'] = test_df['score'].apply(lambda x: np.exp(x)/(1+np.exp(x)))
test_df['target'] = test_df['target'].fillna(0)
test_df[['description_id', 'paper_id', 'target']].to_csv("result_m2_{}_5cv.csv".format(model_id), index=False)


