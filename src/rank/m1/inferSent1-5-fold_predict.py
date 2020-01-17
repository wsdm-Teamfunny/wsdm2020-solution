#!/usr/bin/env python
# coding: utf-8

# In[1]:


#coding=utf-8
########################################
## import packages
########################################
from __future__ import division
import sys
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,GRU,Conv1D,GlobalMaxPool1D,MaxPooling1D,CuDNNGRU,TimeDistributed, Lambda, multiply,concatenate,CuDNNLSTM,Bidirectional
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import sys
import copy
import tensorflow as tf

from sklearn.metrics import roc_auc_score
import pickle

from keras import backend as K
K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES']='0'
np.random.seed(2019)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)

from keras.engine.topology import Layer
# from keras import initializations
from keras import initializers, regularizers, constraints
import random
random.seed(2019)
np.random.seed(2019)


# In[2]:


get_ipython().run_line_magic('connect_info', '')


# In[3]:


get_ipython().run_cell_magic('time', '', 'test_df = pd.read_csv("./recall_test.csv")')


# In[4]:


test_df.head()


# In[5]:


test_df = test_df[test_df['rank']<50]


# In[6]:


test_df.shape


# In[7]:


BATCH_SIZE = 256
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
# EPOCHS = 4
QT_MAX_LEN = 128
QB_MAX_LEN = 128
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


# In[8]:


test_df['corp'] = test_df['corp'].astype(str)
test_df['quer_key'] = test_df['quer_key'].astype(str)


# In[9]:


with open("tokenizer.pkl","rb") as f: #二次训练时不需要
    tokenizer = pickle.load(f)  


# In[10]:


word_index = tokenizer.word_index #二次训练时不需要


# In[11]:


def build_embedding_matrix(word_index, embedding_index):
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    print(nb_words)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    cnt = 0
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS: ####
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        elif word.capitalize() in embedding_index:
            embedding_matrix[i] = embedding_index[word.capitalize()]
        elif word.lower() in embedding_index:
            embedding_matrix[i] = embedding_index[word.lower()]
        elif word.upper() in embedding_index:
            embedding_matrix[i] = embedding_index[word.upper()]
        else:
            embedding_matrix[i] = embedding_index['something']
            cnt+=1
#     del embedding_index
#     gc.collect()
    print("None word:{}".format(cnt))
    return embedding_matrix

def build_embeddings():####使用
    logger.info('Load and build embeddings')
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(word_index, f) for f in embedding_indexs], axis=-1)
    return embedding_matrix
####load_embeddings
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


# In[12]:


get_ipython().run_cell_magic('time', '', "# emb_index_glove = load_embeddings('../input/glove.840B.300d.txt')\n# emb_index_fasttext = load_embeddings('../input/crawl-300d-2M.vec')\nwith open('glove_vec.pkl','rb') as f:\n    emb_index_glove = pickle.load(f)\nwith open('w2v_vec.pkl','rb') as f:\n    emb_index_fasttext = pickle.load(f)")


# In[13]:


emb_matrix_glove = build_embedding_matrix(word_index,emb_index_glove)
emb_matrix_fasttext = build_embedding_matrix(word_index,emb_index_fasttext)
final_embedding_matrix = np.concatenate([emb_matrix_glove,emb_matrix_fasttext], axis=-1)
final_embedding_matrix.shape


# In[14]:


from keras.engine.topology import Layer
class Attention(Layer):
    def __init__(self, step_dim=599,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


# In[15]:


from keras.layers import SpatialDropout1D,GlobalAveragePooling1D,GlobalMaxPooling1D,CuDNNLSTM,LSTM,CuDNNGRU
from keras.layers.merge import add

def s2vector(embedding_matrix,lstm_units=128,dense_hidden_units=128*4):
    input_tensor = Input(shape=(QT_MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_tensor)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(lstm_units, return_sequences=True))(x)
#     x = Dropout(0.1)(x)
#     x = Bidirectional(CuDNNGRU(lstm_units, return_sequences=True))(x)
    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(dense_hidden_units, activation='relu')(hidden)])
    hidden = Dropout(0.1)(hidden)
#     hidden = add([hidden, Dense(dense_hidden_units, activation='relu')(hidden)])
    extract_model = Model(input_tensor,hidden)
    return extract_model

def buildmodel():
    qt = Input(shape=(QT_MAX_LEN,),name='q_t')
    qb = Input(shape=(QB_MAX_LEN,),name='q_b')
    extract_model = s2vector(embedding_matrix=final_embedding_matrix)
#     extract_model.summary()
    x_qt = extract_model(qt)
    x_qb = extract_model(qb)
    
#     cat = concatenate([x_qt,x_qb])
    p_sub_h = Lambda(lambda x: K.abs(x[0] - x[1]))([x_qt, x_qb])
    p_mul_h = multiply([x_qt, x_qb])
    p_concat_h = concatenate([x_qt, x_qb, p_sub_h, p_mul_h])
    p_concat_h = Dense(units=128, activation='relu')(p_concat_h)
    result = Dense(1, activation='sigmoid')(p_concat_h)
    model = Model(inputs=[qt,qb], outputs=result)
    return model


# In[16]:


K.clear_session()
model = buildmodel()
model.summary()
optimizer = keras.optimizers.Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer)


# In[17]:


get_ipython().run_cell_magic('time', '', "qt_test = tokenizer.texts_to_sequences(list(test_df['corp']))\nqb_test = tokenizer.texts_to_sequences(list(test_df['quer_key']))")


# In[18]:


get_ipython().run_cell_magic('time', '', 'qt_test = pad_sequences(qt_test, maxlen=QT_MAX_LEN)\nqb_test = pad_sequences(qb_test, maxlen=QB_MAX_LEN)')


# In[31]:


def predict_one_fold(model,cv,test_df):
    pred_df = test_df[['description_id','paper_id']]
    model_name = "inferSent1"
    model_temp_dir = f"temp/{model_name}"
    model_bst_dir = f"weights/{model_name}"
    model_bst_path = os.path.join(model_bst_dir,f"{model_name}_{cv}_bst.h5")
    model.load_weights(model_bst_path)
    pred_valid = model.predict([qt_test,qb_test],batch_size=512,verbose=1)
    print(pred_valid.shape)
    pred_df['pred'] = pred_valid[:,0]
    pred_df.to_csv(f'../../../output/m1/{model_name}/test_{cv}.csv',index=False)
    return pred_valid


# In[32]:


# del final_embedding_matrix,emb_matrix_glove,emb_matrix_fasttext


# In[33]:


# del tokenizer
# gc.collect()


# In[34]:


# test_df['pred'] = pred_valid


# In[35]:


# test_df.head()


# In[36]:


# test_df['pred'] = 0
for i in range(1,6):
    pred_valid = predict_one_fold(model,i,test_df)
#     test_df['pred'] += pred_valid


# In[ ]:


result = []
for i in range(1,6):
    re_csv = f"../../../output/m1/{model_name}/test_{i}.csv"
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


final_test.to_csv(f"../../../output/m1/{model_name}/te_{model_name}newtest.csv",index=False)

