import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
import matchzoo as mz
from matchzoo.preprocessors.units.truncated_length import TruncatedLength
from utils import MAP, build_matrix, topk_lines, predict, Logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='bert_002')
args = parser.parse_args()

model_id = args.model_id

if model_id=="bert_002":
    test_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_final_test_processed_query_key.dp")
    bst_epochs = {1:1, 2:1, 3:2, 4:1, 5:1}
if model_id=="bert_003":
    test_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_test_processed_query_all.dp")
    bst_epochs = {1:2, 2:1, 3:1, 4:2, 5:1}
if model_id=="bert_004":
    test_processed = mz.data_pack.data_pack.load_data_pack(
        "bert_data/bert_final_test_processed_query_all_nopreprocessing.dp/")
    bst_epochs = {1:2, 2:2, 3:1, 4:1, 5:1}

padding_callback = mz.models.Bert.get_default_padding_callback()
testset = mz.dataloader.Dataset(
    data_pack=test_processed,
    batch_size=128,
    sort=False,
    shuffle=False
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)


num_dup = 1
num_neg = 7

losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
padding_callback = mz.models.Bert.get_default_padding_callback()
task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]

model = mz.models.Bert()

model.params['task'] = task
model.params['mode'] = 'bert-base-uncased'
model.params['dropout_rate'] = 0.2

model.build()

print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6, t_total=-1)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    trainloader=testloader,
    validloader=testloader,
    validate_interval=None,
    epochs=1
)


for fold in range(1,6):
    i = bst_epochs[fold]
    trainer.restore_model("save/{}_fold_{}_epoch_{}.pt".format(model_id, fold, i))

    score = predict(trainer, testloader)
    X, y = test_processed.unpack()
    result = pd.DataFrame(data={
        'description_id': X['id_left'],
        'paper_id': X['id_right'],
        'score': score[:,0]})
    # result.to_csv("result/{}/{}_fold_{}_test.csv".format(model_id, model_id, fold), index=False)
    result.to_csv("result/{}/final_{}_fold_{}_test.csv".format(model_id, model_id, fold), index=False)


