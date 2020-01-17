import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
import matchzoo as mz
from matchzoo.preprocessors.units.truncated_length import TruncatedLength
from utils import MAP, build_matrix, topk_lines, predict, Logger

from matchzoo.data_pack import DataPack

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='bert_002')
args = parser.parse_args()

model_id = args.model_id

num_dup = 1
num_neg = 7

losses = mz.losses.RankCrossEntropyLoss(num_neg=num_neg)
padding_callback = mz.models.Bert.get_default_padding_callback()
task = mz.tasks.Ranking(losses=losses)
task.metrics = [
    mz.metrics.MeanAveragePrecision(),
    MAP()
]

with Logger(log_filename = '{}.log'.format(model_id)):
    for fold in range(1,6):
        if model_id=='bert_002':
            train_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_train_processed_{}.dp".format(fold))
            val_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_val_processed_{}.dp".format(fold))
        if model_id=='bert_003':
            train_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_train_processed_query_all_{}.dp".format(fold))
            val_processed = mz.data_pack.data_pack.load_data_pack("bert_data/bert_val_processed_query_all_{}.dp".format(fold))
        if model_id=='bert_004':
            train_processed = mz.data_pack.data_pack.load_data_pack(
                "bert_data/bert_train_processed_query_all_nopreprocessing_{}.dp".format(fold))
            val_processed = mz.data_pack.data_pack.load_data_pack(
                "bert_data/bert_val_processed_query_all_nopreprocessing_{}.dp".format(fold))

        model = mz.models.Bert()

        model.params['task'] = task
        model.params['mode'] = 'bert-base-uncased'
        model.params['dropout_rate'] = 0.2

        model.build()

        print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


        trainset = mz.dataloader.Dataset(
            data_pack=train_processed,
            mode='pair',
            num_dup=num_dup,
            num_neg=num_neg,
            batch_size=1,
            resample=True,
            sort=False,
            shuffle=True
        )
        trainloader = mz.dataloader.DataLoader(
            dataset=trainset,
            stage='train',
            callback=padding_callback
        )

        valset = mz.dataloader.Dataset(
            data_pack=val_processed,
            batch_size=32,
            sort=False,
            shuffle=False
        )
        valloader = mz.dataloader.DataLoader(
            dataset=valset,
            stage='dev',
            callback=padding_callback
        )


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
            trainloader=trainloader,
            validloader=valloader,
            validate_interval=None,
            epochs=1
        )

        for i in range(0,8):
            print("="*10+" epoch: "+str(i)+" "+"="*10)
            trainer.run()
            trainer.save_model()
            os.rename("save/model.pt", "save/{}_fold_{}_epoch_{}.pt".format(model_id, fold, i))


        