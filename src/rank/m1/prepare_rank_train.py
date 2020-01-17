#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm


# In[2]:


paper = pd.read_feather("../../../input/paper_input_final.ftr")


# In[3]:


paper['abst'] = paper['abst'].apply(lambda s: s.replace('no_content', ''))
paper['corp'] = paper['titl']+' '+paper['keywords'].fillna('').replace(';', ' ')+paper['abst']


# In[4]:


df_train = pd.read_feather("../../../input/tr_input_final.ftr")


# In[5]:


df_train.head()


# In[6]:


df_test = pd.read_feather("../../../input/te_input_final.ftr")


# In[7]:


df_test.head()


# In[8]:


#####reduce mem
import datetime
def pandas_reduce_mem_usage(df):
    start_mem=df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    starttime = datetime.datetime.now()
    for col in df.columns:
        col_type=df[col].dtype   #每一列的类型
        if col_type !=object:    #不是object类型
            c_min=df[col].min()
            c_max=df[col].max()
            # print('{} column dtype is {} and begin convert to others'.format(col,col_type))
            if str(col_type)[:3]=='int':
                #是有符号整数
                if c_min<0:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.iinfo(np.uint8).min and c_max<=np.iinfo(np.uint8).max:
                        df[col]=df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
            #浮点数
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # print('\t\tcolumn dtype is {}'.format(df[col].dtype))

        #是object类型，比如str
        else:
            # print('\t\tcolumns dtype is object and will convert to category')
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    endtime = datetime.datetime.now()
    print('consume times: {:.4f}'.format((endtime - starttime).seconds))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[9]:


recall_train = pd.read_feather('../../../input/tr_s0_32-50.ftr')
recall_test = pd.read_feather('../../../input/te_s0_32-50.ftr')


# In[10]:


recall_train = pandas_reduce_mem_usage(recall_train)


# In[11]:


recall_test = pandas_reduce_mem_usage(recall_test)


# In[12]:


recall_train.shape


# In[13]:


cv_id = pd.read_csv("../../../input/cv_ids_0109.csv")
recall_train.drop(columns=['cv'],axis=1,inplace=True)
recall_train = recall_train.merge(cv_id,on=['description_id'],how='left')


# In[14]:


recall_train = recall_train.dropna(subset=['cv']).reset_index(drop=True)


# In[15]:


recall_train.shape,recall_test.shape


# In[16]:


recall_train = recall_train.merge(paper[['paper_id','corp']],on=['paper_id'],how='left')
recall_test = recall_test.merge(paper[['paper_id','corp']],on=['paper_id'],how='left')


# In[17]:


recall_train = recall_train.merge(df_train[['description_id','quer_key','quer_all']],on=['description_id'],how='left')
recall_test = recall_test.merge(df_test[['description_id','quer_key','quer_all']],on=['description_id'],how='left')


# In[18]:


recall_train = recall_train.sort_values(['description_id', 'corp_sim_score'], ascending=False)
recall_train['rank'] = recall_train.groupby('description_id').cumcount().values
recall_test = recall_test.sort_values(['description_id', 'corp_sim_score'], ascending=False)
recall_test['rank'] = recall_test.groupby('description_id').cumcount().values


# In[19]:


keep_columns = ['description_id','paper_id','corp','quer_key','quer_all','corp_sim_score','cv','rank','target']
recall_train = recall_train[keep_columns].reset_index(drop=True)
recall_test = recall_test[keep_columns].reset_index(drop=True)


# In[20]:


recall_train.head()


# In[22]:


recall_train.to_csv('recall_train.csv',index=False)


# In[23]:


recall_test.to_csv('recall_test.csv',index=False)


# In[ ]:


# recall_train.shape

