
# coding: utf-8

# In[1]:


# external vec
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import time
from datetime import datetime
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import corpora, models, similarities
from gensim.similarities import SparseMatrixSimilarity
from gensim.similarities import MatrixSimilarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# In[3]:


paper = pd.read_feather("../../../input/paper_input_final.ftr")


# In[4]:


paper['abst'] = paper['abst'].apply(lambda s: s.replace('no_content', ''))
paper['corp'] = paper['titl']+' '+paper['keywords'].fillna('').replace(';', ' ')+paper['abst']


# In[5]:


paper.head()


# In[6]:


paper['len'] = paper['corp'].apply(len)


# In[7]:


paper['len'].describe()


# In[8]:


df_train = pd.read_feather("../../../input/tr_input_final.ftr")


# In[9]:


df_train.head()


# In[10]:


df_train['len'] = df_train['quer_key'].apply(len)
df_train['len'].describe()


# In[16]:


df_test = pd.read_feather("../../../input/te_input_final.ftr")


# In[17]:


df_test.head()


# In[18]:


# df_train[df_train['quer_all'].str.contains("[##]")]


# In[19]:


from tqdm import tqdm
###训练语料准备
with open("corpus.txt","w+") as f:
    for i in tqdm(range(len(paper))):
        abst = paper.iloc[i]['abst']
        if abst!='no_content' and abst!="none":
            f.write(abst+"\n")
        title = paper.iloc[i]['titl']
        if title!='no_content' and title!="none":
            f.write(title+"\n")
    for i in tqdm(range(len(df_train))):
        quer_all = df_train.iloc[i]['quer_all']
        f.write(quer_all+"\n")
    for i in tqdm(range(len(df_test))):
        quer_all = df_test.iloc[i]['quer_all']
        f.write(quer_all+"\n")


# In[23]:


####word2vector
from gensim.models import word2vec
sentences = word2vec.LineSentence('./corpus.txt') 
model = word2vec.Word2Vec(sentences, sg=1,min_count=2,window=8,size=300,iter=6,sample=1e-4, hs=1, workers=12)  


# In[24]:


model.save("word2vec.model")


# In[34]:


model.wv.save_word2vec_format("word2vec.txt",binary=False)


# In[26]:


#glove的已有
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


# In[31]:


# 输入文件
glove_file = datapath('glove/vectors.txt')
# 输出文件
tmp_file = get_tmpfile("glove_vec.txt")

