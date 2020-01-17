#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
pd.options.display.precision = 15
pd.set_option('max_colwidth',200)
import lightgbm as lgb
import time
import datetime

np.random.seed(1234)
import warnings
warnings.filterwarnings("ignore")
import gc


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


import os
import numpy as np
import pandas as pd
import librosa
import pickle
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
import os
import tensorflow as tf
import random

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from keras.layers import Dense, Input, Activation
from keras.layers import BatchNormalization,Add,Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import callbacks
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import keras
import warnings
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)
import os


# In[5]:


np.random.seed(1234)


# In[6]:


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        if col == 'scalar_coupling_constant':continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[7]:


feat_dir = "../../../feat/"
input_dir = "../../../input/"


# In[8]:


cv_id = pd.read_csv("../../../input/cv_ids_0109.csv")


# In[9]:


# cv_id.head()


# In[10]:


train = pd.read_feather(f'{feat_dir}/tr_s0_32-50.ftr')
train.drop(columns=['cv'],axis=1,inplace=True)
train = train.merge(cv_id,on=['description_id'],how='left')


# In[11]:


train.shape


# In[12]:


train = train.dropna(subset=['cv']).reset_index(drop=True)


# In[13]:


train.shape


# In[14]:


test = pd.read_feather(f'{feat_dir}/te_s0_32-50.ftr')


# In[15]:


test2 = pd.read_feather(f'{feat_dir}/te_s0_20-50.ftr') ####只是为了归一化，可以在训练时保存Scaler


# In[16]:


train.head()


# In[17]:


test.head()


# In[18]:


ID_NAMES = ['description_id', 'paper_id']
TARGET_NAME = 'target'
def get_feas(data):
    cols = data.columns.tolist()
    del_cols = ID_NAMES + ['target', 'cv']
    #sub_cols = ['year', 'corp_cos', 'corp_eucl', 'corp_manh', 'quer_all']
    sub_cols = ['year', 'corp_sim_score']
    sub_cols = ['year', 'pos_of_corp', 'pos_of_abst', 'pos_of_titl']
    toxic_cols = ['quer_key_tfidf_abst_cos_dis','quer_key_tfidf_titl_cos_dis','quer_key_tfidf_corp_cos_dis']
    for col in data.columns:
        for sub_col in sub_cols:
            if sub_col in col:
                del_cols.append(col)
                
    del_cols += toxic_cols
    cols = [val for val in cols if val not in del_cols]
    print ('del_cols', del_cols)
    return cols


# In[19]:


feature_columns = get_feas(train)


# In[20]:


# feature_columns = list(set(all_columns)-set(id_columns+target_columns+not_use_columns))


# In[21]:


len(feature_columns)


# In[22]:


len_train = len(train)


# In[25]:


train_test = pd.concat([train,test2])


# In[26]:


# train_test['corp_sim_score'].max()


# In[27]:


from tqdm import tqdm
for fea in tqdm(feature_columns):
    if train_test[fea].max()<=1.0:continue
    else:
        scaler = MinMaxScaler() ##StandardScaler()
        train_test[fea] = scaler.fit(train_test[fea].values.reshape(-1,1))
        train[fea] = scaler.transform(train[fea].values.reshape(-1,1))
        test[fea] = scaler.transform(test[fea].values.reshape(-1,1))


# In[28]:


# train = train_test[:len_train]
# test = train_test[len_train:]
del train_test
gc.collect()


# In[29]:


train.head()


# In[30]:


test.head()


# In[31]:


def create_nn_model(input_shape):
    inp = Input(shape=(input_shape,))
    bn_inp = BatchNormalization()(inp)
    x = Dense(512)(bn_inp)
#     x = Dense(200)(inp)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x0 = Dropout(0.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x1 = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x = keras.layers.add([x,x1])
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x2 = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x = keras.layers.add([x,x2,x0])
    #x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x3 = Dropout(0.2)(x)
    x = Dense(128)(x3)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization()(x)
    x4 = keras.layers.add([x,x3])
    x = Dense(64)(x4)
    x = LeakyReLU(alpha=0.05)(x)
    x = keras.layers.concatenate([x,x4])
#     x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)#scalar_coupling_constant
    model = Model(inputs=inp, outputs=out)
    model.summary()
    return model


# In[32]:


train['cv'] = train['cv'].astype(int)


# In[33]:


# train = train.sample(frac=1).reset_index(drop=True)


# In[34]:


len(train[train['target']==0])/len(train[train['target']==1])


# In[35]:


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


# In[36]:


model_dir = "./m1_model/nn02/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# In[37]:


tr_data = pd.read_csv(f'{input_dir}/train_release.csv')
tr_data = tr_data[['description_id', 'paper_id']].rename(columns={'paper_id': 'target_id'})


# In[38]:


CV_RESULT_OUT=True


# In[39]:


# feature_columns[95:100]


# In[40]:


# used_feas = [
# #  'corp_sim_score',
# #  'quer_key_tfidf_abst_cos_dis',###toxic
# #  'quer_key_tfidf_abst_eucl_dis',
# #  'quer_key_tfidf_abst_manh_dis',
# # #  'quer_key_tfidf_titl_cos_dis',###toxic
    
# #  'quer_key_tfidf_titl_eucl_dis',
# #  'quer_key_tfidf_titl_manh_dis',
# # #  'quer_key_tfidf_corp_cos_dis', ###toxic
    
# #  'quer_key_tfidf_corp_eucl_dis',
# #  'quer_key_tfidf_corp_manh_dis',
# #  'quer_all_tfidf_abst_cos_dis',
# #  'quer_all_tfidf_abst_eucl_dis',
# #  'quer_all_tfidf_abst_manh_dis',
# #  'quer_all_tfidf_titl_cos_dis',
# #  'quer_all_tfidf_titl_eucl_dis',
# #  'quer_all_tfidf_titl_manh_dis',
# #  'quer_all_tfidf_corp_cos_dis',
# #  'quer_all_tfidf_corp_eucl_dis',
# #  'quer_all_tfidf_corp_manh_dis',
# #  'quer_key_corp_bm25_score',
# #  'quer_key_abst_bm25_score',
# #  'quer_key_titl_bm25_score',
# #  'quer_all_corp_bm25_score',
# #  'quer_all_abst_bm25_score',
# #  'quer_all_titl_bm25_score'
# ]


# In[41]:


# !rm -rf ../model/nn02/
# !mkdir ../model/nn02


# In[42]:


used_feas = feature_columns


# In[43]:


from keras.models import Model, load_model
def train_one_fold(type_train_df,type_test_df,model_dir,cv,pi=False):
    print(" fold " + str(cv))
    train_data = type_train_df[(type_train_df['cv']!=cv)]
#     train_data_pos = train[train['target']==1]
#     train_data_neg = train[train['target']==0].sample(frac=0.1)
#     train_data = pd.concat([train_data_pos,train_data_neg]).reset_index(drop=True)
    valid_data = type_train_df[(type_train_df['cv']==cv)]
    des_id = valid_data['description_id']
    paper_id = valid_data['paper_id']
    train_x = train_data[used_feas].values
    train_y = train_data['target'].values
    valid_x = valid_data[used_feas].values
    valid_y = valid_data['target'].values
    
    test_x = type_test_df[used_feas].values
    
    input_shape = train_x.shape[1]
    print(input_shape)
    if True: ###只做训练
        model = create_nn_model(input_shape)
        opt = keras.optimizers.Nadam(lr=3e-4)
        model.compile(optimizer=opt, loss='binary_crossentropy')
        best = [-1, 0, 0, 0]  # score, epoch, cv_result, result
        earlystop = 20
        es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=20,verbose=1, mode='auto', restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75,patience=10, min_lr=1e-6, mode='auto', verbose=1)
        model_name = "fold_{}_best.model".format(str(cv))
        model_name_wrt = os.path.join(model_dir,model_name)
        sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='val_loss', save_best_only=True, period=1)
        if not os.path.exists(model_name_wrt):
            print("model not found")
#             history = model.fit(train_x,train_y, 
#                 validation_data=(valid_x,valid_y), 
#                 callbacks=[es, rlr, sv_mod], epochs=1000, batch_size=2048, verbose=1)
        else:
            try:
#                 K.clear_session()
#                 K.set_learning_phase(1)
                model = load_model(model_name_wrt)
            except Exception as e:
                print(e)
#                 history = model.fit(train_x,train_y, 
#                     validation_data=(valid_x,valid_y), 
#                     callbacks=[es, rlr, sv_mod], epochs=1000, batch_size=2048, verbose=1)
        cv_predict=model.predict(valid_x,verbose=1)
        cv_score_fold = cal_map(cv_predict,cv,type_train_df,tr_data)
    else:
        model_name = "xxxx自己选的model"
    if CV_RESULT_OUT:
        cv_preds = cv_predict
        rdf = pd.DataFrame()
        rdf = rdf.reindex(columns=['description_id','paper_id','pred'])
        rdf['description_id'] = des_id
        rdf['paper_id'] = paper_id
        rdf['pred'] = cv_preds
    test_des_id = type_test_df['description_id']
    test_paper_id = type_test_df['paper_id']
    test_preds = model.predict(test_x,batch_size=2048)
    test_df = pd.DataFrame()
    test_df = test_df.reindex(columns=['description_id','paper_id','pred'])
    test_df['description_id'] = test_des_id
    test_df['paper_id'] = test_paper_id
    test_df['pred'] = test_preds
    return rdf,test_df,cv_score_fold


# In[44]:


# train[train['target']==1].describe()


# In[45]:


kfold = 5
type_scores = []
type_cv_results = []
type_test_results = []
model_name = '../../../output/m1/nn02/nn02'
fold_scores = []
fold_cv_results = []
fold_test_results = []
# test_preds = np.zeros(len(test))
for cv in range(1,kfold+1):#####这里是因为cv是1~5
    cv_df,test_df,cv_score = train_one_fold(train,test,model_dir,cv)
#     fold_cv_results.append(cv_df)
#     fold_test_results.append(test_df)
    cv_df.to_csv(f"{model_name}_cv_{cv}.csv",index=False)
    test_df.to_csv(f"{model_name}_result_{cv}.csv",index=False)
    print("fold {} finished".format(cv))
    print(cv_score)
    fold_scores.append(cv_df)


# In[46]:


result = []
for i in range(1,6):
    re_csv = f"../../../output/m1/nn02/nn02_result_{i}.csv"
    test_df = pd.read_csv(re_csv)
    result.append(test_df)


# In[47]:


final_test = result[0].copy()


# In[48]:


final_test.head()


# In[49]:


for i in range(1,5):
    final_test['pred']+=result[i]['pred']


# In[50]:


final_test['pred'] = final_test['pred']/5


# In[51]:


# input_dir = "../../../input/"
# test_df = pd.read_csv(f"{input_dir}/validation.csv")


# In[52]:


final_test.to_csv("../../../output/m1/nn02/te_nn02newtest.csv",index=False)


# In[49]:


# df_pred = final_test


# In[50]:


# sort_df_pred = df_pred.sort_values(['description_id', 'pred'], ascending=False)
# df_pred = df_pred[['description_id']].drop_duplicates() \
#     .merge(sort_df_pred, on=['description_id'], how='left')
# df_pred['rank'] = df_pred.groupby('description_id').cumcount().values
# df_pred = df_pred[df_pred['rank'] < 3]
# df_pred = df_pred.groupby(['description_id'])['paper_id'] \
#     .apply(lambda s : ','.join((s))).reset_index()

# df_pred = test_df[['description_id']].drop_duplicates().merge(df_pred, on=['description_id'], how='left')


# In[51]:


# df_pred.head()


# In[52]:


# df_pred.dropna(inplace=True)


# In[53]:


# df_pred.shape


# In[55]:


# from tqdm import tqdm
# fo = open("../result/nn_02/sub_nn02.csv", 'w+')
# for i in tqdm(range(df_pred.shape[0])):
#     desc_id = df_pred.iloc[i]['description_id']
#     paper_ids = df_pred.iloc[i]['paper_id']
# #     paper_ids = paper_ids.replace("\t",",")
#     print (str(desc_id) + ',' + paper_ids, file=fo)
# fo.write(",55a38fe7c91b587b095b0d1c,55a4eb3e65ceb7cb02dbff7c,55a3a74065ce5cd7b3b2db98")
# fo.close()


# In[ ]:




