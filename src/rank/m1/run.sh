#!/bin/sh

####依赖paper_input_1.ftr、te_input_1.ftr
####写入语料，并训练word2vector词向量
python3 w2v_training.py ###暂时用jupyter notebook占位

###训练glove词向量
cd glove && make
bash demo.sh
####回到目录
cd ..

###词向量序列化



###准备训练数据
python3 prepare_rank_train.py ###暂时用jupyter notebook占位

###inferSent-simple-5-fold训练
python3 inferSent1-5-fold_train.py ###暂时用jupyter notebook 占位

###inferSent-simple-5-fold预测
python3 inferSent1-5-fold_predict.py ###暂时用jupyter notebook 占位

###catboost模型训练&预测
python3 catboost3.py ###暂时用jupyter notebook 占位

###nn02模型训练
python3 nn02_train.py ###暂时用jupyter notebook 占位
python3 nn02_predict.py ###暂时用jupyter notebook 占位
