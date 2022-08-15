# -*- coding: utf-8 -*-
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
import os


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# group 별 sort 후 파일 저장
# path = './data/df2.csv'
# df = pd.read_csv(path, parse_dates=['updated'],encoding='utf-8', low_memory=False)
# devName = df['dev_name'].to_list()
# # print(set(devName))
# # print(len(list(set(devName))))
# # print(df.drop_duplicates(['dev_name'], keep='first').groupby(by=['dev_id'], as_index=False).size().sum())
#
#
#
#
# nameList = ['101호', '102호', '103호', '1층공용', '201호', '202호', '203호(고시텔)', '2층공용1', '301호', '302호', '303호', '304호', '305호', '306호', '307호', '308호', '309호', '310호', '311호', '312호', '3층공용', '401호', '402호', '403호', '404호', '405호', '406호', '407호', '408호', '409호', '410호', '411호', '412호', '4층공용', '501호', '502호', '503호', '504호', '505호', '506호', '507호', '508호', '509호', '510호', '511호', '512호', '5층공용', '601호', '602호', '603호', '604호', '605호', '606호', '607호', '608호', '609호', '610호', '611호', '612호', '6층공용', '701호', '702호', '703호', '704호', '705호', '706호', '707호', '708호', '709호', '710호', '711호', '712호', '7층공용', '801호', '802호', '803호', '804호', '805호', '806호', '807호', '808호', '809호', '810호', '8층공용', '8층환경', '8층환경1', '8층환경2', '수팰리스8층환경1', '수팰리스8층환경2']
#
# cnt = [] #계측 횟수
# medi = [] # 중위 증가량
# maxDiff = [] # 최대 증가량
# minDiff = []# 최소 증가량
# meanDiff = []# 평균 증가량
#
#
#
# for name in nameList :
#     cnt.append(df[df['dev_name'] == name].count()[0])
#     medi.append(df[df['dev_name'] == name]["power_value"].diff().median())
#     maxDiff.append(df[df['dev_name'] == name]["power_value"].diff().max())
#     minDiff.append(df[df['dev_name'] == name]["power_value"].diff().min())
#     meanDiff.append(df[df['dev_name'] == name]["power_value"].diff().mean())
#
#     # print(name,'의 계측 횟수 : ' , df[df['dev_name'] == name].count()[0])
#     # print(name,'의 중위 증가량 : ' , df[df['dev_name'] == name]["power_value"].diff().median())
#     # print(name,'의 최대 증가량 : ' , df[df['dev_name'] == name]["power_value"].diff().max())
#     # print(name,'의 최소 증가량 : ' , df[df['dev_name'] == name]["power_value"].diff().min())
#     # print(name,'의 평균 증가량 : ' , df[df['dev_name'] == name]["power_value"].diff().mean())
#
# diffFrame = pd.DataFrame([cnt, medi, maxDiff, minDiff, meanDiff], columns=nameList)
# print(diffFrame.head(10))
# diffFrame.to_csv('./data/diffFrame.csv')

# group 별 sort 후 파일 저장
path = './data/diffFrame.csv'
df = pd.read_csv(path, encoding='utf-8', low_memory=False)
df = df[['101호', '102호', '103호', '1층공용', '201호', '202호', '203호(고시텔)', '2층공용1', '301호', '302호', '303호', '304호', '305호', '306호', '307호', '308호', '309호', '310호', '311호', '312호', '3층공용', '401호', '402호', '403호', '404호', '405호', '406호', '407호', '408호', '409호', '410호', '411호', '412호', '4층공용', '501호', '502호', '503호', '504호', '505호', '506호', '507호', '508호', '509호', '510호', '511호', '512호', '5층공용', '601호', '602호', '603호', '604호', '605호', '606호', '607호', '608호', '609호', '610호', '611호', '612호', '6층공용', '701호', '702호', '703호', '704호', '705호', '706호', '707호', '708호', '709호', '710호', '711호', '712호', '7층공용', '801호', '802호', '803호', '804호', '805호', '806호', '807호', '808호', '809호', '810호', '8층공용', '8층환경', '8층환경1', '8층환경2', '수팰리스8층환경1', '수팰리스8층환경2']]
df = df.T
df.columns = ['cnt','medi','maxDiff','minDiff','meanDiff']

print(df.head(12))


print(df.max(axis=0)[0]) # 계측횟수
print(df.max(axis=0)[1]) # 중위 증가량
print(df.max(axis=0)[2]) # 최대 증가량
print(df.max(axis=0)[3])# 최소 증가량
print(df.max(axis=0)[4])# 평균 증가량