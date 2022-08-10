import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch.nn as nn
import torch
import torch.optim as optim
import pickle
import sys


pd.set_option('display.max_columns', None)

''' 학습/테스트 데이터 분할 '''
# 2년 간의 데이터
path = './data/tb_meter_log201901-202201_all_columns_2.csv'

df = pd.read_csv(path, parse_dates=['updated'], encoding='utf-8', low_memory=False)
df.sort_values(by=['updated'], inplace=True)
df.set_index('updated', inplace=True)



print(df.info())

print(df.head(50))
print(df.tail(50))


scaler = MinMaxScaler()
df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))



sys.exit()

df.to_csv('./data/tb_meter_total_sortDate.csv')
sys.exit()


#print(df.info())
df2 = df['updated','dev_id','dev_name','power_value']


print(df.head(50))
print(df.tail(50))