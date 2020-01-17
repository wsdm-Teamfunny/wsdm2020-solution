import numpy as np
import pandas as pd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='ESIMplus_001')
args = parser.parse_args()

model_id = args.model_id

stk_path = "../../../stk_feat"

df = pd.read_csv("oof_m2_{}_5cv.csv".format(model_id))
df = df.rename(columns={"target": "pred"})
df.to_feather("{}/m2_{}_tr.ftr".format(stk_path, model_name))

df = pd.read_csv("result_m2_{}_5cv.csv".format(model_id))
df = df.rename(columns={"target": "pred"})
df.to_feather("{}/final_m2_{}_te.ftr".format(stk_path, model_name))

