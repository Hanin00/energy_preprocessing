import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import datetime

'''
    개선에 관한 의견
    1. 누적 값이고, 각 3분 정도의 간격이 있다고 해도 항상 같은게 아니니까
        누적 값 간 증가량을 특징으로 하면 더 잘 나올 듯(빈 값들은 보간하고)
'''

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#데이터 불러오기
df = pd.read_csv('./data/target102_3H_diff.csv',parse_dates=['updated'],  encoding = 'utf-8', )
df.set_index('updated', inplace=True)

#결측치 있어서 보간 필요(index를 datatime으로 해서 그런지는 모름 이유 파악 X)
df_intp_linear = df.interpolate()
data_ski = df_intp_linear[["power_value","pw_diff"]]

## scaling
scaler = MinMaxScaler()
data_ski["power_value"] = scaler.fit_transform(data_ski["power_value"].values.reshape(-1, 1))
data_ski["pw_diff"] = scaler.fit_transform(data_ski["pw_diff"].values.reshape(-1, 1))

#window_size = 학습시 고려하는 이전 일자
## sequence data
def make_dataset(data, window_size=7):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(data.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)



