#!/usr/bin/env python$
# -*- coding: utf-8 -*-$

# 数据处理
import numpy as np
import pandas as pd
import feather

# 基础文件读写
def load_df(filename, nrows=None):
    if filename.endswith('csv'):
        return pd.read_csv(filename, nrows = nrows)
    elif filename.endswith('ftr'):
        return feather.read_dataframe(filename)

def save_df(df, filename, index=False):
    if filename.endswith('csv'):
        df.to_csv(filename, index=index)
    elif filename.endswith('ftr'):
        df = df.reset_index(drop=True)
        df.columns = [str(col) for col in df.columns]
        df.to_feather(filename)

# merge 特征文件
def merge_fea(df_list, primary_keys=[]):
    assert len(primary_keys) >= 0, 'empty primary keys'
    print (df_list)

    df_base = load_df(df_list[0])
    for i in range(1, len(df_list)):
        print (df_list[i])
        cur_df = load_df(df_list[i])
        df_base = pd.concat([df_base, \
                cur_df.drop(primary_keys, axis=1)], axis=1)
    print ('merge completed, df shape', df_base.shape)
    return df_base

# 模型预测结果输出
def out_preds(target_name, df_ids, ypreds, out_path, labels=[]):
    preds_df = pd.DataFrame(df_ids)
    preds_df[target_name] = ypreds
    if len(labels) == preds_df.shape[0]:
        preds_df['label'] = np.array(labels)
    elif len(labels) > 0:
        print ('labels length not match')
    preds_df.to_csv(out_path, float_format = '%.4f', index=False)

#def out_preds(id_name, target_name, ids, ypreds, out_path, labels=[]):
#    preds_df = pd.DataFrame({id_name: np.array(ids)})
#    preds_df[target_name] = ypreds
#    if len(labels) == preds_df.shape[0]:
#        preds_df['label'] = np.array(labels)
#    elif len(labels) > 0:
#        print ('labels length not match')
#    preds_df.to_csv(out_path, float_format = '%.4f', index=False)




