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
pd.set_option('display.max_rows', None)
#
# ''' 통합 데이터 row num, updated로 인덱스 지정하고 시간 순 정렬 '''
# # path = './data/tb_meter_log201901-202201_all_columns_2.csv'
# # df = pd.read_csv(path, parse_dates=['updated'], encoding='utf-8', low_memory=False)
# # df.sort_values(by=['updated'], inplace=True)
# # df.set_index('updated', inplace=True)
# #
# # #cnt,dev_sid,dev_id,meter_id,dev_name,dev_location,power_value,power_value2,meterType,gas_value,water_value,hwater_value,heat_value,updated,magnitudeValue,power_child,parent,ptf_value,pth_value,temp_t1,temp_t2,iptf_value,ipth_value
#
# scaler = MinMaxScaler()
# df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))
# df2 = df[['dev_sid','dev_id','meter_id','dev_name','dev_location', 'power_value' ]]
# df2.to_csv('./data/tb_meter_sorted.csv')


# path = './data/tb_meter_sorted.csv'
#
# df = pd.read_csv(path, parse_dates=['updated'], encoding='utf-8', low_memory=False)
# df["pw_diff"] = df["power_value"].diff()
# df2 = df[['updated','dev_id','dev_sid', 'dev_name',"power_value"]]
# #
# # print(df2.groupby('dev_id').size())
# # print(df2.groupby(['dev_id','dev_name']).size())
#
# # 가구별 데이터 건수
# #devNameCnt = df2.groupby(['dev_sid','dev_name']).size()
#
# df2.to_csv("./data/df2.csv")
# sys.exit()



pd.set_option('display.max_columns', None)
# group 별 sort 후 파일 저장
path = './data/df2.csv'
df = pd.read_csv(path, parse_dates=['updated'],encoding='utf-8', low_memory=False)

print(df['dev_name']=='101호'.sort_values(['updated'],ascending=True).head(5))










#print(df.sort_values(['dev_id','dev_sid','updated'],ascending=True).groupby('dev_sid').head(5))

df4 =  pd.DataFrame(df.sort_values(['dev_id','dev_sid','updated'],ascending=True).groupby('dev_sid')[['dev_id','dev_sid','dev_name','power_value']]).reset_index()
df4.to_csv("./data/groupby.csv")
sys.exit()

df = df.groupby(by=['dev_sid'], as_index=False)
print(df.head(10))

sys.exit()



df3 = df2[['dev_id','dev_name']]

print(df3.groupby(by=['dev_id','dev_name'], as_index=False).size())
df3Cnt = df3.groupby(by=['dev_id','dev_name'], as_index=False).size()['size'].tolist()


print("가구별 데이터 수")
df4 = pd.DataFrame(df3Cnt)

print("평균 : ",df4.mean())
print("중위값 : ",df4.median())
print("최대 : ",df4.max())
print("최소 : ",df4.min())





sys.exit()




print(df2.groupby(by=['dev_id'], as_index=False).size(), '\n') #기기별 데이터
print("기기별 데이터 수 max")
print(df2.groupby(by=['dev_id'], as_index=False).size().max(), '\n') #기기별 데이터
print("기기별 데이터 수 min",)
print(df2.groupby(by=['dev_id'], as_index=False).size().min(), '\n') #기기별 데이터
#print("기기별 데이터 수 mean")
df2Cnt = df2.groupby(by=['dev_id'], as_index=False).size()['size'].tolist()
print("기기별 데이터 수 mean", sum(df2Cnt)/len(df2Cnt)) #기기별 데이터

print("기기별 데이터수")
df4 = pd.DataFrame(df2Cnt)

print("평균 : ",df4.mean())
print("중위값 : ",df4.median())
print("최대 : ",df4.max())
print("최소 : ",df4.min())


sys.exit()


print(df3.groupby(by=['dev_id'], as_index=False).value_counts()) #기기별 데이터
print(df3.groupby(by=['dev_id'], as_index=False).value_counts().mean()) #기기별 데이터


sys.exit()



#print(df3.groupby(by=['dev_id'], as_index=False).size())

print("0001")
print(df3.drop_duplicates(['dev_name'], keep='first')) # 기기별 가구
print("0002")
print(df3.drop_duplicates(['dev_name'], keep='first').groupby(by=['dev_id'], as_index=False))
print("0003")
print(df3.drop_duplicates(['dev_name'], keep='first').groupby(by=['dev_id'], as_index=False).size())
print("0004")
print(df3.drop_duplicates(['dev_name'], keep='first').groupby(by=['dev_id'], as_index=False).size().mean())
print("0005")
print(df3.groupby(by=['dev_id'], as_index=False).value_counts().mean()) #기기별 데이터
print("0006")
print(df3.groupby(by=['dev_name'], as_index=False).value_counts().mean()) #가구별 데이터


print(df3.groupby(by=['dev_id'], as_index=False).value_counts().agg())
print(df3.groupby(by=['dev_id','dev_name'], as_index=False).value_counts().agg())
sys.exit()


#
#
#
# print(df3.groupby(by=['dev_id'], as_index=False).size()) # 기기당 데이터 수
# print(df3.groupby(by=['dev_id'], as_index=False)['dev_name'].value_counts().agg())
# print(df3.groupby(by=['dev_id'], as_index=False).value_counts().agg())
#
# sys.exit()
#
#
#
# print(df3.groupby(by=['dev_id'], as_index=False).size().agg({ "dfsize ": [min, max, sum]}))
#
#
#
#
# print('기기별, 가구별 데이터 수')
# # print(df2.groupby(by=['dev_id'], as_index=False).count())
# print(df2.groupby(by=['dev_id'], as_index=False).size()) # 기기당 데이터 수
# print(df2.groupby(by=['dev_id'], as_index=False).size().agg({"": [min, max, sum]}))
# print(df2.groupby(by=['dev_id'], as_index=False).value_counts())
# print(df2.groupby(by=['dev_id'], as_index=False).value_counts()).agg()
# # print(df2.groupby(by=['dev_id','dev_sid'], as_index=False).count())
# print(df2.groupby(by=['dev_id','dev_sid'], as_index=False).value_counts())
#
# df3 = df2[['dev_id','dev_name']]
# print('df3 기기별 통계')
# print(df3.groupby(by=['dev_id'], as_index=False).value_counts()).agg()
# print(df3.groupby(by=['dev_id'], as_index=False).count())
#

# print(devNameCnt.iloc[:-1].sum()) # 마지막 열만
# print(devNameCnt.iloc[:-1].median()) # 마지막 열만
# print(devNameCnt.iloc[:-1].mean()) # 마지막 열만
# print(devNameCnt.iloc[:-1].max()) # 마지막 열만






# devName = df['dev_name'].to_list()  # 가구 명
# print(len(list(set(devName))))  #89
# print('devName[:20] : ', list(set(devName))[:20]) # devName[:20] :  ['310호', '503호', '409호', '504호', '8층공용', '604호', '412호', '307호', '505호', '605호', '306호', '705호', '802호', '6층공용', '712호', '806호', '603호', '710호', '706호', '608호']
#
# devSid = df['dev_sid'].to_list()  # 계량기가 달린 각 가구의 수?
# print('  : ',len(list(set(devSid))))  # 86
# print('devSid[:20] : ',list(set(devSid))[:20])  #devSid[:20] :  [169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188]
#
# devId = df['dev_id'].to_list()  # 계량기 대수 2019030005가 하나 더 껴있음.  select distinct mac from tb_meter; 와 다른 점
# print(len(list(set(devId))))  # 11
# print('devId[:20] : ',  list(set(devId))[:20])  # devId[:20] :  ['B0AF201809030007', 'B0AF201809030008', 'B0AF201901030005', 'B0AF201809030004', 'B0AF201809030005', 'B0AF201809030001', 'B0AF201809030002', 'B0AF201902250002', 'B0AF201809140018', 'B0AF201809030003', 'B0AF201809030006']
#
#
#

