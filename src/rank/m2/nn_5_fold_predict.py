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

model_id = args.model_id

num_dup = 6
num_neg = 10
batch_size = 128
add_lgb_feat = False
debug = False

if model_id == 'ESIMplus_001':
    bst_epochs = {1:0, 2:2, 3:4, 4:2, 5:1}
    Model = ESIMplus
    lr = 0.001
    add_lgb_feat = True
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}


if model_id == 'aNMM_001':
    bst_epochs = {1:4, 2:4, 3:3, 4:4, 5:9}
    Model = mz.models.aNMM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0, 
              'dropout_rate': 0.1}
    
if model_id == 'ESIM_001':
    bst_epochs = {1:4, 2:4, 3:2, 4:2, 5:6}
    Model = mz.models.ESIM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}
    
if model_id == 'MatchLSTM_001':
    bst_epochs = {1:4, 2:2, 3:2, 4:4, 5:3}
    Model = mz.models.MatchLSTM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0}

losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]

if model_id == 'ESIM_001_pointwise':
    bst_epochs = {1:4, 2:3, 3:7, 4:12, 5:5}
    Model = mz.models.ESIM
    lr = 0.001
    params = {'embedding_freeze': True,
              'mask_value': 0,
              'lstm_layer': 2, 
              'hidden_size': 200, 
              'dropout': 0.2}
    
    task = mz.tasks.Classification(num_classes=2)
    task.metrics = ['acc']
    
    
padding_callback = Model.get_default_padding_callback()
embedding_matrix = np.load("data/embedding_matrix.npy")
# l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
# embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

test_processed = mz.data_pack.data_pack.load_data_pack("test_processed.dp")
testset = mz.dataloader.Dataset(
    data_pack=test_processed,
    batch_size=batch_size,
    sort=False,
    shuffle=False
)

testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)



model = Model()
if add_lgb_feat: model.set_feature_dim(30)

model.params['task'] = task
model.params['embedding'] = embedding_matrix

for param in params:
    model.params[param] = params[param]

model.build()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = mz.trainers.Trainer(
            model=model,
            optimizer=optimizer,
            trainloader=testloader,
            validloader=testloader,
            validate_interval=None,
            epochs=1
)


for fold in range(1,6):
    i = bst_epochs[fold]
    val_processed = mz.data_pack.data_pack.load_data_pack("5fold/val_processed_{}.dp".format(fold))
    valset = mz.dataloader.Dataset(
        data_pack=val_processed,
        batch_size=batch_size,
        sort=False,
        shuffle=False
    )

    valloader = mz.dataloader.DataLoader(
        dataset=valset,
        stage='dev',
        callback=padding_callback
    )

    trainer.restore_model("save/{}_fold_{}_epoch_{}.pt".format(model_id, fold, i))

    score = predict(trainer, valloader)
    X, y = val_processed.unpack()
    result = pd.DataFrame(data={
        'description_id': X['id_left'],
        'paper_id': X['id_right'],
        'score': score[:,0]})
    result.to_csv("result/{}/{}_fold_{}_cv.csv".format(model_id, model_id, fold), index=False)

    score = predict(trainer, testloader)
    X, y = test_processed.unpack()
    result = pd.DataFrame(data={
        'description_id': X['id_left'],
        'paper_id': X['id_right'],
        'score': score[:,0]})
    result.to_csv("result/{}/{}_fold_{}_test.csv".format(model_id, model_id, fold), index=False)
    
    
    