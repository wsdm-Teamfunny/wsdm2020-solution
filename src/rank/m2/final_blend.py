import numpy as np 
import pandas as pd
from tqdm import tqdm

np.set_printoptions(precision=4)

def map3_func(df, topk = 50, verbose=0):
    ids = df[df.label==1].description_id.values
    df_recalled = df[df.description_id.isin(ids)].reset_index(drop=True)
    df_recalled = df_recalled.sort_values(
        by=['description_id', 'label'], ascending=False).reset_index(drop=True)
    result = df_recalled.score.values.reshape([-1,topk])
    ranks = topk-result.argsort(axis=1).argsort(axis=1)
    map3_sum = sum(((1/ranks[:,0])*(ranks[:,0]<4)))
    if verbose>0:
        print("recall rate: "+str((df_recalled.shape[0]/topk)/(df.shape[0]/topk)))
        print("map@3 in recall: "+str(map3_sum/(df_recalled.shape[0]/topk)))
    print("map@3 in all: "+str(map3_sum/(df.shape[0]/topk)))
    
    
m2_path = "../../model/"

res = pd.read_feather('{}/lgb_s0_m2_33-0/lgb_s0_m3_33.ftr'.format(m2_path))
res['score'] = res['target'].apply(lambda x:np.log(x/(1-x)))
res.loc[res['score']<-12, 'score'] = -12
res = res[['description_id', 'paper_id', 'score']]
res.head()

res1 = pd.read_feather('{}/lgb_s0_m2_33-1/lgb_s0_m3_33.ftr'.format(m2_path))
res1['score'] = res1['target'].apply(lambda x:np.log(x/(1-x)))
res1.loc[res1['score']<-12, 'score'] = -12
res1 = res1[['description_id', 'paper_id', 'score']]
res1.head()


res2 = pd.read_feather('{}/lgb_s0_m3_34-0/lgb_s0_m3_34.ftr'.format(m2_path))
res2['score'] = res2['target'].apply(lambda x:np.log(x/(1-x)))
res2.loc[res2['score']<-12, 'score'] = -12
res2 = res2[['description_id', 'paper_id', 'score']]
res2.head()


res3 = pd.read_feather('{}/lgb_s0_m3_34-1/lgb_s0_m3_34.ftr'.format(m2_path))
res3['score'] = res3['target'].apply(lambda x:np.log(x/(1-x)))
res3.loc[res3['score']<-12, 'score'] = -12
res3 = res3[['description_id', 'paper_id', 'score']]
res3.head()


res4 = pd.read_feather('{}/lgb_s0_m3_35-0/lgb_s0_m3_35.ftr'.format(m2_path))
res4['score'] = res4['target'].apply(lambda x:np.log(x/(1-x)))
res4.loc[res4['score']<-12, 'score'] = -12
res4 = res4[['description_id', 'paper_id', 'score']]
res4.head()


res5 = pd.read_feather('{}/lgb_s0_m3_35-1/lgb_s0_m3_35.ftr'.format(m2_path))
res5['score'] = res5['target'].apply(lambda x:np.log(x/(1-x)))
res5.loc[res5['score']<-12, 'score'] = -12
res5 = res5[['description_id', 'paper_id', 'score']]
res5.head()


res6 = pd.read_feather('{}/lgb_s0_m3_38-0lgb_s0_m3_38.ftr'.format(m2_path))
res6['score'] = res6['target'].apply(lambda x:np.log(x/(1-x)))
res6.loc[res6['score']<-12, 'score'] = -12
res6 = res6[['description_id', 'paper_id', 'score']]
res6.head()


res7 = pd.read_feather('{}/lgb_s0_m3_38-1/lgb_s0_m3_38.ftr'.format(m2_path))
res7['score'] = res7['target'].apply(lambda x:np.log(x/(1-x)))
res7.loc[res7['score']<-12, 'score'] = -12
res7 = res7[['description_id', 'paper_id', 'score']]
res7.head()


res8 = pd.read_feather('{}/lgb_s0_m3_40-0/lgb_s0_m3_40.ftr'.format(m2_path))
res8['score'] = res8['target'].apply(lambda x:np.log(x/(1-x)))
res8.loc[res8['score']<-12, 'score'] = -12
res8 = res8[['description_id', 'paper_id', 'score']]
res8.head()


res9 = pd.read_feather('{}/model/m1/m1_catboost13.ftr'.format(m2_path))
res9['score'] = res9['pred'].apply(lambda x:np.log(x/(1-x)))
res9.loc[res9['score']<-12, 'score'] = -12
res9 = res9[['description_id', 'paper_id', 'score']]
res9.head()


model_id = 'bert_002'
res_b1 = pd.read_csv("final_result_m2_{}_5cv.csv".format(model_id))
res_b1['score'] = res_b1['target'].apply(lambda x:np.log(x/(1-x)))
res_b1.loc[res_b1['score']<-12, 'score'] = -12
res_b1 = res_b1[['description_id', 'paper_id', 'score']]
res_b1.head()


model_id = 'bert_003'
res_b2 = pd.read_csv("final_result_m2_{}_5cv.csv".format(model_id))
res_b2['score'] = res_b2['target'].apply(lambda x:np.log(x/(1-x)))
res_b2.loc[res_b2['score']<-12, 'score'] = -12
res_b2 = res_b2[['description_id', 'paper_id', 'score']]
res_b2.head()


model_id = 'bert_004'
res_b3 = pd.read_csv("final_result_m2_{}_5cv.csv".format(model_id))
res_b3['score'] = res_b3['target'].apply(lambda x:np.log(x/(1-x)))
res_b3.loc[res_b3['score']<-12, 'score'] = -12
res_b3 = res_b3[['description_id', 'paper_id', 'score']]
res_b3.head()

model_id = 'bert_year_test'
res_b4 = pd.read_csv("final_result_m2_{}_5cv.csv".format(model_id))
res_b4['score'] = res_b4['target'].apply(lambda x:np.log(x/(1-x)))
res_b4.loc[res_b4['score']<-12, 'score'] = -12
res_b4 = res_b4[['description_id', 'paper_id', 'score']]
res_b4.head()


res_all = res.rename(columns={'score': 'score_0'}).merge(
    res1.rename(columns={'score': 'score_1'}), how='outer', on=['description_id', 'paper_id']).merge(
    res2.rename(columns={'score': 'score_2'}), how='outer', on=['description_id', 'paper_id']).merge(
    res3.rename(columns={'score': 'score_3'}), how='outer', on=['description_id', 'paper_id']).merge(
    res4.rename(columns={'score': 'score_4'}), how='outer', on=['description_id', 'paper_id']).merge(
    res5.rename(columns={'score': 'score_5'}), how='outer', on=['description_id', 'paper_id']).merge(
    res6.rename(columns={'score': 'score_6'}), how='outer', on=['description_id', 'paper_id']).merge(
    res7.rename(columns={'score': 'score_7'}), how='outer', on=['description_id', 'paper_id']).merge(
    res8.rename(columns={'score': 'score_8'}), how='outer', on=['description_id', 'paper_id']).merge(
    res9.rename(columns={'score': 'score_9'}), how='outer', on=['description_id', 'paper_id']).merge(
    res_b1.rename(columns={'score': 'score_b1'}), how='outer', on=['description_id', 'paper_id']).merge(
    res_b2.rename(columns={'score': 'score_b2'}), how='outer', on=['description_id', 'paper_id']).merge(
    res_b3.rename(columns={'score': 'score_b3'}), how='outer', on=['description_id', 'paper_id']).merge(
    res_b4.rename(columns={'score': 'score_b4'}), how='outer', on=['description_id', 'paper_id'])
res_all = res_all.fillna(0.0)
res_all.head()


cols = ['score_0', 'score_1', 'score_2', 'score_3', 'score_4', 'score_5', 
        'score_6', 'score_7', 'score_8', 'score_9',
        'score_b1', 'score_b2', 'score_b3']

corr_matrix = []
for description_id, df_tmp in tqdm(res_all.groupby('description_id')):
    corr_matrix.append(
        df_tmp[cols].corr().values[:,:,np.newaxis])
corr_matrix = np.concatenate(corr_matrix, axis=2)
corr_matrix[np.isnan(corr_matrix)] = 0
pd.DataFrame(data=corr_matrix.mean(axis=2), columns=cols, index=cols)

res_all['score'] = (
    (
        res_all['score_0'] + res_all['score_1'] + res_all['score_2'] + res_all['score_3'] + 
        res_all['score_4'] + res_all['score_5'] + res_all['score_6'] + res_all['score_7']
    )/8 + 
    (
        res_all['score_8'] + res_all['score_9']
    )/2 +
    (
        res_all['score_b1'] + 1.5*res_all['score_b2']
    )/2.5*5 + 
    (
        res_all['score_b2'] + 3*res_all['score_b3']
    )/4
)


result = res_all.sort_values(by=['description_id', 'score'], na_position='first').groupby(
    'description_id').tail(3)


model_id = 'all_model'

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
sub.to_csv("blend_{}.csv".format(model_id), header=False, index=False)
print("blend_{}.csv".format(model_id))

