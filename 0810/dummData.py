# LSTM - https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM
# Arima Ex - https://wikidocs.net/50949
import sys
import numpy as np
import random
import pandas as pd
from datetime import datetime
import math, time
import itertools
import datetime
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler




df1D = pd.read_csv('../data/total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
dateColumn = 'updated'
freq = 'D'


def slicerFreq(df, freq, dateColumn) :
    df.set_index(dateColumn, inplace=True)
    resultDf = df.resample(freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf["power_value"] = df_intp_linear[["power_value"]]
    return resultDf


resultDf = slicerFreq(df1D, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간
#min, max 값은 normalize 하기 전 값을 가지고 있어야 함





print(resultDf.tail(40))



y_max_pre_normalize = 0.9795547554696371
y_min_pre_normalize = 0.0

# y_max_pre_normalize = max(resultDf["power_value"])
# y_min_pre_normalize = min(resultDf["power_value"])

# print(y_max_pre_normalize)
# print(y_min_pre_normalize)
# print(resultDf.head(10))


scaler = MinMaxScaler()
resultDf["power_value"] = scaler.fit_transform(resultDf["power_value"].values.reshape(-1, 1))

print(resultDf[:-1])

def denormalize(y):
    final_value = y*(y_max_pre_normalize - y_min_pre_normalize) + y_min_pre_normalize
    return final_value


# 2019-01-03일 데이터 norm 값 / 2019-01-03     0.675184
print("denorm : ",denormalize(0.9346875))  #denorm 값 :0.675184...


predDenorm = 0.9346875
predDenorm -= 0.000001
print("index : ",  list(resultDf.index[resultDf["power_value"] >= predDenorm])[0])

















'''
    denormalize 예제
'''
# 
# # 일단  y_train 값에서 마지막 값을 얻을 수 있는지 확인 하고, 만약 맞으면,  pred 값을 추가
# # normalized 값이 아닌 원래 powervalue로 denormalize 함.
# predLast = y_train.detach().numpy()[-2] #train Dataset의 normalize 된 마지막 값
# predLast -= 0.000001 # float이라 정확한 값 비교가 어려워서 부등호를 이용함. 이때 사용하기 위해 값을 임의로 줄임
# 
# #초기의 dataset은 trainset을 기반으로 만들어 짐
# 
# test_max_pre_normalize = max(ARTrainset)
# test_min_pre_normalize = min(ARTrainset)
# 
# trainset = pd.DataFrame(ARTrainset, columns=["power_value"])
# scaler = MinMaxScaler()
# trainset["power_value"] = scaler.fit_transform(trainset["power_value"].values.reshape(-1, 1))
# testFirDenorm = denormalize(predLast,test_max_pre_normalize, test_min_pre_normalize) #denormalize 된 원래 의 값
# trainFirDenorm = denormalize(predLast,train_max_pre_normalize, train_min_pre_normalize) #denormalize 된 원래 의 값
# origin_max_pre_normalize = max(resultDf["power_value"])
# origin_min_pre_normalize = min(resultDf["power_value"])
# totalDenorm = denormalize(trainFirDenorm, origin_max_pre_normalize, origin_min_pre_normalize) # 마지막 날짜에 대한 예측
# 
























# '''
#     originTSetDf : Origin Test Set, Dataframe
#     datalen : XL의 데이터 개수
#     newPv : 바로 앞에서 예측한 Power_value의 Denormalize 값
# '''
#
# def appendNewData(originTSetDt, dataLen, newPv) :
#     td_pd = pd.Timedelta(1, unit='days')
#     nextIdx = pd.Timestamp(pd.Timestamp(pd.to_datetime(originTSetDt.index[-1].date()) + td_pd).date())
#     resultDf.loc[str(nextIdx)] = [newPv]
#     testXDf = resultDf[::-1]
#     testXDf = testXDf[:dataLen]
#     return testXDf
#
# newPv = 3.1212
# resultDf = appendNewData(resultDf, 28, newPv)
#
# print(resultDf)
# print(len(resultDf))