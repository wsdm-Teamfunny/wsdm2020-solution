#!/usr/bin/env python
#coding=utf-8

# 基础模块
import math
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

# 数据处理
import numpy as np
import pandas as pd

def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expls):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    for col_expl in col_expls:
        df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    base_cols = list(set(df.columns) - set(col_expls))
    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expls[0]].str.len())
         for col in base_cols}
    )

    for col_expl in col_expls:
        df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
        df_out.loc[:, col_expl] = df_out[col_expl]
    return df_out
