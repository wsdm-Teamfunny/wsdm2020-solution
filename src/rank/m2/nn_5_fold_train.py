import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import matchzoo as mz
from model import ESIMplus

from utils import MAP, build_matrix, topk_lines, predict, Logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='ESIMplus_001')
args = parser.parse_args()

num_dup = 6
num_neg = 10
batch_size = 128
add_lgb_feat = False
debug = False

if model_id == 'ESIMplus_001':
    Model = ESIMplus
    lr = 0.001
    add_lgb_feat = True
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}


if model_id == 'aNMM_001':
    Model = mz.models.aNMM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0, 
              'dropout_rate': 0.1}
    
if model_id == 'ESIM_001':
    Model = mz.models.ESIM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}
    
if model_id == 'MatchLSTM':
    model_id = 'MatchLSTM_001'
    Model = mz.models.MatchLSTM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0}

losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
padding_callback = Model.get_default_padding_callback()
task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]

if model_id == 'ESIM_001_pointwise':
    Model = mz.models.ESIM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}
    
    task = mz.tasks.Classification(num_classes=2)
    task.metrics = ['acc']

embedding_matrix = np.load("data/embedding_matrix.npy")


if not os.path.exists('result/{}'.format(model_id)):
    os.makedirs('result/{}'.format(model_id))

with Logger(log_filename = '{}.log'.format(model_id)):
    for fold in range(1,5):
        print("="*10+" fold: "+str(fold)+" data_processed prepare "+"="*10)
        train_processed = mz.data_pack.data_pack.load_data_pack("5fold/train_processed_{}.dp".format(fold))
        val_processed = mz.data_pack.data_pack.load_data_pack("5fold/val_processed_{}.dp".format(fold))
        
        if model_id == 'ESIM_001_pointwise':
            train_processed.relation.label = train_processed.relation.label.astype(np.long)
            val_processed.relation.label = val_processed.relation.label.astype(np.long)


        print("="*10+" fold: "+str(fold)+" dataset prepare "+"="*10)
        trainset = mz.dataloader.Dataset(
            data_pack=train_processed,
            mode='pair',
            num_dup=num_dup,
            num_neg=num_neg,
            batch_size=batch_size,
            resample=True,
            sort=False,
            shuffle=True
        )
        valset = mz.dataloader.Dataset(
            data_pack=val_processed,
            batch_size=batch_size,
            sort=False,
            shuffle=False
        )

        print("="*10+" fold: "+str(fold)+" dataloader prepare "+"="*10)
        trainloader = mz.dataloader.DataLoader(
            dataset=trainset,
            stage='train',
            callback=padding_callback
        )
        valloader = mz.dataloader.DataLoader(
            dataset=valset,
            stage='dev',
            callback=padding_callback
        )

        print("="*10+" fold: "+str(fold)+" model build "+"="*10)
        model = Model()
        if add_lgb_feat: model.set_feature_dim(30)

        model.params['task'] = task
        model.params['embedding'] = embedding_matrix

        for param in params:
            model.params[param] = params[param]

        model.build()
        if debug: print(model)

        print("="*10+" fold: "+str(fold)+" trainers build "+"="*10)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        trainer = mz.trainers.Trainer(
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            validloader=valloader,
            validate_interval=None,
            epochs=1
        )

        print("="*10+" fold: "+str(fold)+" training "+"="*10)
        trainer.restore_model("save/{}_fold_{}_epoch_{}.pt".format(model_id, fold, 1))
        for i in range(2,6):
            trainer._model.embedding.requires_grad_(requires_grad=False)
            print("="*10+" fold: "+str(fold)+" epoch: "+str(i)+" "+"="*10)
            trainer.run()
            trainer.save_model()
            os.rename("save/model.pt", "save/{}_fold_{}_epoch_{}.pt".format(model_id, fold, i))
            
            
