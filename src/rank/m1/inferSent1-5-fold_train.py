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




#读数据
train_df = pd.read_csv("recall_train.csv")
train_df = train_df.dropna(subset=['cv']).reset_index(drop=True)
test_df = pd.read_csv("../../input/recall_test.csv") ###二次训练的时候不需要读取test_df




y = train_df[['target']].values

train_df.shape


BATCH_SIZE = 256
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
# EPOCHS = 4
QT_MAX_LEN = 128
QB_MAX_LEN = 128
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300


train_df['corp'] = train_df['corp'].astype(str)
train_df['quer_key'] = train_df['quer_key'].astype(str)
test_df['corp'] = test_df['corp'].astype(str)
test_df['quer_key'] = test_df['quer_key'].astype(str)


# In[9]:


# %%time
tokenizer = Tokenizer()
#分词转id

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='',lower=False)  #二次训练时不需要
tokenizer.fit_on_texts(list(train_df['corp'])
                      +list(train_df['quer_key'])
                      +list(test_df['corp']
                      +list(test_df['quer_key']))
                      )


# with open("tokenizer.pkl","rb") as f: #二次训练时需要
#     tokenizer = pickle.load(f)  


word_index = tokenizer.word_index #二次训练时不需要




len(word_index)



# %%time
with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f,pickle.HIGHEST_PROTOCOL)


qt_train = tokenizer.texts_to_sequences(list(train_df['corp'])) ###
qb_train = tokenizer.texts_to_sequences(list(train_df['quer_key']))

# ####pad
qt_train = pad_sequences(qt_train, maxlen=QT_MAX_LEN)
qb_train = pad_sequences(qb_train, maxlen=QB_MAX_LEN)


# with open("qt_train.pkl","rb") as f: #二次训练
#     qt_train = pickle.load(f)
# with open("qb_train.pkl","rb") as f:
#     qb_train = pickle.load(f)



with open("qt_train.pkl","wb") as f:
    pickle.dump(qt_train,f,pickle.HIGHEST_PROTOCOL)
with open("qb_train.pkl","wb") as f:
    pickle.dump(qb_train,f,pickle.HIGHEST_PROTOCOL)


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


# In[18]:

emb_index_glove = load_embeddings('glove_vec.txt')
emb_index_fasttext = load_embeddings('word2vec.txt')

import pickle
with open("glove_vec.pkl","wb") as f:
    pickle.dump(emb_index_glove, f, pickle.HIGHEST_PROTOCOL)
with open("w2v_vec.pkl","wb") as f:
    pickle.dump(emb_index_fasttext, f, pickle.HIGHEST_PROTOCOL)


emb_matrix_glove = build_embedding_matrix(word_index,emb_index_glove)
emb_matrix_fasttext = build_embedding_matrix(word_index,emb_index_fasttext)
final_embedding_matrix = np.concatenate([emb_matrix_glove,emb_matrix_fasttext], axis=-1)
final_embedding_matrix.shape



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


# In[23]:


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



train_df = train_df.reset_index(drop=True)

tr_data = pd.read_csv('../../../input/train_release.csv')
tr_data = tr_data[['description_id', 'paper_id']].rename(columns={'paper_id': 'target_id'})

import ml_metrics as metrics
def cal_map(pred_valid,cv,description_ids,paper_ids,tr_data):
    df_pred = pd.DataFrame({'description_id':description_ids,
                           'paper_id':paper_ids})
    df_pred['pred'] = pred_valid
    df_pred = df_pred[['description_id','paper_id','pred']]
    sort_df_pred = df_pred.sort_values(['description_id', 'pred'], ascending=False)
    df_pred = df_pred[['description_id']].drop_duplicates().merge(sort_df_pred, on=['description_id'], how='left')
    df_pred['rank'] = df_pred.groupby('description_id').cumcount().values
    df_pred = df_pred[df_pred['rank'] < 3]
    df_pred = df_pred.groupby(['description_id'])['paper_id'].apply(lambda s : ','.join((s))).reset_index()
    df_pred = df_pred.merge(tr_data, on=['description_id'], how='left')
    df_pred.rename(columns={'paper_id': 'paper_ids'}, inplace=True)
    df_pred['paper_ids'] = df_pred['paper_ids'].apply(lambda s: s.split(','))
    df_pred['target_id'] = df_pred['target_id'].apply(lambda s: [s])
    return metrics.mapk(df_pred['target_id'].tolist(), df_pred['paper_ids'].tolist(), 3)


# In[27]:


def train_one_fold(cv,Training=True):
    cv=cv
    y = train_df[['target']].values
    idx_train = train_df[train_df['cv'] != cv].index
    idx_val = train_df[train_df['cv'] == cv].index
    train_des_ids =  train_df[train_df['cv'] != cv]['description_id'].values
    train_paper_ids = train_df[train_df['cv'] != cv]['paper_id'].values

    valid_des_ids = train_df[train_df['cv'] == cv]['description_id'].values
    valid_paper_ids = train_df[train_df['cv']==cv]['paper_id'].values

    labels_train = y[idx_train]
    labels_valid = y[idx_val]
    cv_qt_train = qt_train[idx_train]
    cv_qt_valid = qt_train[idx_val]

    cv_qb_train = qb_train[idx_train]
    cv_qb_valid = qb_train[idx_val]
    model_name = "inferSent1"
    model_temp_dir = f"temp/{model_name}"
    model_bst_dir = f"weights/{model_name}"
    if not os.path.exists(model_temp_dir):
        os.makedirs(model_temp_dir)
    if not os.path.exists(model_bst_dir):
        os.makedirs(model_bst_dir)

    bst_models = []
    bst_scores = []
    ###fold data

    best = [-1, 0, 0]  # socre, epoch, cv_result
    earlystop = 5
    lr_patience = 2
    min_lr = 1e-5
    lr = 1e-3
    no_improve_lr = 0
    K.clear_session()
    model = buildmodel()
    model.summary()
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model_bst_path = os.path.join(model_bst_dir,f"{model_name}_{cv}_bst.h5")
    try:
        model.load_weights(model_bst_path)
    except:
        pass
    for epoch in range(1000):
        if not Training:
            break
        model_path = os.path.join(model_temp_dir,f"{model_name}_{cv}_{epoch}.h5")####快照集成
        model.fit([cv_qt_train,cv_qb_train],labels_train,batch_size=1024, epochs=1, verbose=1)
        pred_valid = model.predict([cv_qt_valid,cv_qb_valid],batch_size=512,verbose=1)
        s = cal_map(pred_valid,cv,valid_des_ids,valid_paper_ids,tr_data)
        print(epoch,s)
        if s > best[0]:# the bigger is better
            print("epoch " + str(epoch) + " improved from " + str(best[0]) + " to " + str(s))
            best = [s,epoch,pred_valid]
            model.save_weights(model_bst_path)
            no_improve_lr = 0
        lr = float(K.get_value(model.optimizer.lr))
        if epoch > 1 and epoch <2: 
            K.set_value(model.optimizer.lr, 3e-4)
        if epoch>2 and no_improve_lr > lr_patience and  lr > min_lr:
            lr = float(K.get_value(model.optimizer.lr))
            lr = 0.75*lr
            K.set_value(model.optimizer.lr, lr)
            print("Setting lr to {}".format(lr))
            no_improve_lr = 0
        if epoch-best[1]>earlystop:
            break
        no_improve_lr += 1
        model.save_weights(model_path)
    
    bst_models.append(model_bst_path)
    bst_scores.append(best[0])
    model.load_weights(model_bst_path)
    pred_valid = model.predict([cv_qt_valid,cv_qb_valid],batch_size=512,verbose=1)
    s = cal_map(pred_valid,cv,valid_des_ids,valid_paper_ids,tr_data)
    print("cv:{},best score{:.4f}".format(cv,s))
    cv_df = pd.DataFrame({'description_id':valid_des_ids,
                         'paper_ids':valid_paper_ids})
    cv_df['pred']=pred_valid
    cv_df.to_csv(f'../../../output/m1/{model_name}_cv_{cv}.csv',index=False)


# In[28]:


train_one_fold(1,Training=False)


# In[29]:


train_one_fold(2,Training=False)


# In[30]:


train_one_fold(3,Training=False)


# In[31]:


train_one_fold(4,Training=False)


# In[32]:


train_one_fold(5,Training=False)


# In[ ]:




