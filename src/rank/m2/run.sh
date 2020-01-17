#!/bin/sh

python3 preprocessing.py --query_type quer_key
python3 preprocessing.py --query_type quer_all

git clone http://github.com/stanfordnlp/glove
cp gen_w2v.sh glove/
cp data/corpus.txt glove/
cd glove && make
. gen_w2v.sh
cd ..
cp glove/glove.w2v data/

python3 nn_preprocessing.py
python3 bert_preprocessing.py --preprocessing_type fine --left_truncated_length 64 --query_type query_key
python3 bert_preprocessing.py --preprocessing_type fine --left_truncated_length 200 --query_type query_all
python3 bert_preprocessing.py --preprocessing_type coarse --left_truncated_length 200 --query_type query_all

python3 nn_5_fold_train.py --model_id ESIM_001
python3 nn_5_fold_train.py --model_id ESIMplus_001
python3 nn_5_fold_train.py --model_id aNMM_001
python3 nn_5_fold_train.py --model_id MatchLSTM_001
python3 nn_5_fold_train.py --model_id ESIM_001_pointwise

python3 bert_5_fold_train.py --model_id bert_002
python3 bert_5_fold_train.py --model_id bert_003
python3 bert_5_fold_train.py --model_id bert_004

python3 nn_5_fold_predict.py --model_id ESIM_001
python3 nn_5_fold_predict.py --model_id ESIMplus_001
python3 nn_5_fold_predict.py --model_id aNMM_001
python3 nn_5_fold_predict.py --model_id MatchLSTM_001
python3 nn_5_fold_predict.py --model_id ESIM_001_pointwise

python3 bert_5_fold_predict.py --model_id bert_002
python3 bert_5_fold_predict.py --model_id bert_003
python3 bert_5_fold_predict.py --model_id bert_004

python3 fold_result_integration.py --model_id ESIM_001
python3 fold_result_integration.py --model_id ESIMplus_001
python3 fold_result_integration.py --model_id aNMM_001
python3 fold_result_integration.py --model_id MatchLSTM_001
python3 fold_result_integration.py --model_id ESIM_001_pointwise
python3 fold_result_integration.py --model_id bert_002
python3 fold_result_integration.py --model_id bert_003
python3 fold_result_integration.py --model_id bert_004

python3 mk_submission.py --model_id ESIM_001
python3 mk_submission.py --model_id ESIMplus_001
python3 mk_submission.py --model_id aNMM_001
python3 mk_submission.py --model_id MatchLSTM_001
python3 mk_submission.py --model_id ESIM_001_pointwise
python3 mk_submission.py --model_id bert_002
python3 mk_submission.py --model_id bert_003
python3 mk_submission.py --model_id bert_004

python3 change_formatting4stk.py --model_id ESIM_001
python3 change_formatting4stk.py --model_id ESIMplus_001
python3 change_formatting4stk.py --model_id aNMM_001
python3 change_formatting4stk.py --model_id MatchLSTM_001
python3 change_formatting4stk.py --model_id ESIM_001_pointwise
python3 change_formatting4stk.py --model_id bert_002
python3 change_formatting4stk.py --model_id bert_003
python3 change_formatting4stk.py --model_id bert_004

###### finally #####
python3 final_blend.py

