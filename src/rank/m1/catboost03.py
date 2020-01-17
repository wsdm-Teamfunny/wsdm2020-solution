
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
from time import time
from tqdm import tqdm_notebook as tqdm


# In[2]:


feat_dir = "../../../feat/"
input_dir = "../../../input/"
cv_id = pd.read_csv("../../../input/cv_ids_0109.csv")


# In[3]:


train = pd.read_feather(f'{feat_dir}/tr_s0_32-50.ftr')
train.drop(columns=['cv'],axis=1,inplace=True)
train = train.merge(cv_id,on=['description_id'],how='left')
train = train.dropna(subset=['cv']).reset_index(drop=True)
# test = pd.read_feather(f'{feat_dir}/te_s0_20-50.ftr')
test = pd.read_feather(f'{feat_dir}/te_s0_32-50.ftr')


# In[4]:


ID_NAMES = ['description_id', 'paper_id']
TARGET_NAME = 'target'


# In[5]:


def get_feas(data):
    cols = data.columns.tolist()
    del_cols = ID_NAMES + ['target', 'cv']
    #sub_cols = ['year', 'corp_cos', 'corp_eucl', 'corp_manh', 'quer_all']
    sub_cols = ['year', 'corp_sim_score']
    sub_cols = ['year', 'pos_of_corp', 'pos_of_abst', 'pos_of_titl']
    for col in data.columns:
        for sub_col in sub_cols:
            if sub_col in col:
                del_cols.append(col)

    cols = [val for val in cols if val not in del_cols]
    print ('del_cols', del_cols)
    return cols


# In[6]:


feas = get_feas(train)


# In[7]:


def make_classifier():
    clf = CatBoostClassifier(
                               loss_function='Logloss',
                               eval_metric="AUC",
#                                task_type="CPU",
                               learning_rate=0.1, ###0.01
                               iterations=2500, ###2000
                               od_type="Iter",
#                                depth=8,
                               thread_count=10,
                               early_stopping_rounds=100, ###100
    #                            l2_leaf_reg=1,
    #                            border_count=96,
                               random_seed=42
                              )
        
    return clf


# In[8]:


# 开源工具包
import ml_metrics as metrics
def cal_map(pred_valid,cv,train_df,tr_data):
    df_pred = train_df[train_df['cv']==cv].copy()
    df_pred['pred'] = pred_valid
    df_pred = df_pred[['description_id','paper_id','pred']]
    sort_df_pred = df_pred.sort_values(['description_id', 'pred'], ascending=False)
    df_pred = df_pred[['description_id']].drop_duplicates()             .merge(sort_df_pred, on=['description_id'], how='left')
    df_pred['rank'] = df_pred.groupby('description_id').cumcount().values
    df_pred = df_pred[df_pred['rank'] < 3]
    df_pred = df_pred.groupby(['description_id'])['paper_id']             .apply(lambda s : ','.join((s))).reset_index()
    df_pred = df_pred.merge(tr_data, on=['description_id'], how='left')
    df_pred.rename(columns={'paper_id': 'paper_ids'}, inplace=True)
    df_pred['paper_ids'] = df_pred['paper_ids'].apply(lambda s: s.split(','))
    df_pred['target_id'] = df_pred['target_id'].apply(lambda s: [s])
    return metrics.mapk(df_pred['target_id'].tolist(), df_pred['paper_ids'].tolist(), 3)


# In[9]:


import os
model_dir = "./m1_model/catboost03"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# In[10]:


tr_data = pd.read_csv(f'{input_dir}/train_release.csv')
tr_data = tr_data[['description_id', 'paper_id']].rename(columns={'paper_id': 'target_id'})


# In[13]:


for fea in feas:
    if fea not in test.columns:
        print(fea)


# In[14]:


CV_RESULT_OUT=True


# In[15]:


def train_one_fold(type_train_df,type_test_df,model_dir,cv,pi=False):
    print(" fold " + str(cv))
    train_data = type_train_df[(type_train_df['cv']!=cv)]
    valid_data = type_train_df[(type_train_df['cv']==cv)]
    
    des_id = valid_data['description_id']
    paper_id = valid_data['paper_id']
    
    idx_train = train_data.index
    idx_val = valid_data.index
    des_id = valid_data['description_id']
    paper_id = valid_data['paper_id']
    model_name = "fold_{}_cbt_best.model".format(str(cv))
    model_name_wrt = os.path.join(model_dir,model_name)
    clf = make_classifier()
    imp=pd.DataFrame()
    if not os.path.exists(model_name_wrt):
        clf.fit(train_data[feas], train_data[['target']], eval_set=(valid_data[feas],valid_data[['target']]),
                              use_best_model=True, verbose=100)
        clf.save_model(model_name_wrt)
        fea_ = clf.feature_importances_
        fea_name = clf.feature_names_
        imp = pd.DataFrame({'name':fea_name,'imp':fea_})
    else:
        clf.load_model(model_name_wrt)
    cv_predict=clf.predict_proba(valid_data[feas])[:,1]
#     print(cv_predict.shape)
    cv_score_fold = cal_map(cv_predict,cv,type_train_df,tr_data)
    if CV_RESULT_OUT:
        cv_preds = cv_predict
        rdf = pd.DataFrame()
        rdf = rdf.reindex(columns=['description_id','paper_id','pred'])
        rdf['description_id'] = des_id
        rdf['paper_id'] = paper_id
        rdf['pred'] = cv_preds
    test_des_id = type_test_df['description_id']
    test_paper_id = type_test_df['paper_id']
    test_preds = clf.predict_proba(type_test_df[feas])[:,1]
    test_df = pd.DataFrame()
    test_df = test_df.reindex(columns=['description_id','paper_id','pred'])
    test_df['description_id'] = test_des_id
    test_df['paper_id'] = test_paper_id
    test_df['pred'] = test_preds
    return rdf,test_df,cv_score_fold,imp


# In[16]:


kfold = 5
type_scores = []
type_cv_results = []
type_test_results = []
model_name = '../../../output/m1/catboost03/'
fold_scores = []
fold_cv_results = []
fold_test_results = []
imps=[]
# test_preds = np.zeros(len(test))
for cv in range(1,kfold+1):#####这里是因为cv是1~5
    cv_df,test_df,cv_score,imp = train_one_fold(train,test,model_dir,cv)
#     fold_cv_results.append(cv_df)
#     fold_test_results.append(test_df)
    cv_df.to_csv(f"{model_name}_cv_{cv}.csv",index=False)
    test_df.to_csv(f"{model_name}_result_{cv}.csv",index=False)
    imp.to_csv(f"{model_name}_imp_{cv}.csv",index=False)
    print("fold {} finished".format(cv))
    print(cv_score)
    fold_scores.append(cv_score)
    imps.append(imp)


# In[1]:


np.mean(fold_scores)

#0.35309347230573923
#0.3522860689007414
#0.3585175465159315
#0.35720084429290466
#0.34729405401751007


# In[ ]:


result = []
for i in range(1,6):
    re_csv = f"{model_name}_result_{i}.csv"
    test_df = pd.read_csv(re_csv)
    result.append(test_df)


# In[ ]:


final_test = result[0].copy()


# In[ ]:


for i in range(1,5):
    final_test['pred']+=result[i]['pred']


# In[ ]:


final_test['pred'] = final_test['pred']/5


# In[ ]:


final_test.to_csv("../../../output/m1/nn02/te_catboost03newtest.csv",index=False)

