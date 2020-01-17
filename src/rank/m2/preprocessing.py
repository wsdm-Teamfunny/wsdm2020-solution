from tqdm import tqdm
import numpy as np
import pandas as pd
import feather

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--query_type', type=str, default='query_key')
args = parser.parse_args()

query_type = args.query_type

def topk_lines(df, k):
    print(df.shape)
    df.loc[:, 'rank'] = df.groupby(['description_id', 'type']).cumcount().values
    df = df[df['rank'] < k]
    df.drop(['rank'], axis=1, inplace=True)
    print(df.shape)
    return df


## preprocess
candidate_dic = feather.read_dataframe('../../../input/paper_input_final.ftr')

candidate_dic.loc[candidate_dic['keywords'].isna(),'keywords'] = ''
candidate_dic.loc[candidate_dic['titl'].isna(),'titl'] = ''
candidate_dic.loc[candidate_dic['abst'].isna(),'abst'] = ''

candidate_dic['text_right'] = candidate_dic['abst'].str.cat(
    candidate_dic['keywords'], sep=' ').str.cat(
    candidate_dic['titl'], sep=' ')

candidate_dic = candidate_dic.rename(columns={'paper_id': 'id_right'})[['id_right', 'text_right']]
candidate_dic.to_feather('data/candidate_dic.ftr')

train_description = feather.read_dataframe('../../../input/tr_input_final.ftr')

train_description = train_description.rename(
    columns={'description_id': 'id_left', query_type: 'text_left'})
train_description[['id_left', 'text_left']].to_feather('data/train_description_{}.ftr'.format(query_type))


test_description = feather.read_dataframe('../../../input/te_input_final.ftr')

test_description = test_description.rename(
    columns={'description_id': 'id_left', query_type: 'text_left'})
 
test_description[['id_left', 'text_left']].to_feather('data/test_description_{}.ftr'.format(query_type))

train_recall = feather.read_dataframe('../../../feat/tr_s0_32-50.ftr')

## recall
train_recall = train_recall.rename(
    columns={'description_id': 'id_left', 'paper_id': 'id_right', 'target': 'label'})

train_recall = train_recall[train_recall.id_left.isin(train_description.id_left.values)].reset_index(drop=True)
train_recall = train_recall.drop_duplicates()
train_recall = train_recall.fillna(0)
train_recall.to_feather('data/train_recall.ftr')

test_recall = feather.read_dataframe('../../../feat/te_s0_32-50.ftr')
test_recall = test_recall.reset_index(drop=True)

test_recall = test_recall.rename(
    columns={'description_id': 'id_left', 
             'paper_id': 'id_right', 
             'target': 'label'})

# test_recall[['id_left', 'id_right', 'label']].to_feather('data/test_recall.ftr')
test_recall[['id_left', 'id_right', 'label']].to_feather('data/final_test_recall.ftr')


## corpus
if query_type== 'query_key':
    candidate_dic = feather.read_dataframe('data/candidate_dic.ftr')
    train_description = feather.read_dataframe('data/train_description.ftr')
    test_description = feather.read_dataframe('data/test_description.ftr')

    with open('data/corpus.txt','a') as fid:
        for sent in tqdm(candidate_dic['text_right']):
            if type(sent)==str:
                fid.write(sent+'\n')
        for sent in tqdm(train_description['text_left']):
            if type(sent)==str:
                fid.write(sent+'\n')
        for sent in tqdm(test_description['text_left']):
            if type(sent)==str:
                fid.write(sent+'\n')

