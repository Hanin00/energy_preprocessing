import sys
import numpy as np
import random
import pandas as pd
from pylab import mpl, plt
from datetime import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import Util3 as ut

pd.set_option('display.max_columns', None)
ARTrainset = pd.read_pickle("./data/ARTrainset.pkl")
# 데이터 불러오기(10분 단위 데이터)
df1D = pd.read_csv('./data/new_total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
dateColumn = 'updated'
freq = 'D'

resultDf = ut.slicerFreq(df1D, freq, dateColumn)  # 일 단위 데이터로 변환 및 결측치 선형 보간

dateTerm = 28
predS = '2021-08-01'
lenXm = 7
lenXl = 28

#Daily Predict Input
arimaPth ="./model/arima_model_fit.pt"
lstmPth ="./model/LSTM_model.pt"

dailyPredS = '2021-07-04'
dailyPredE = '2021-08-01'

testYs, testXs,  testXm, testXl = ut.testDataTrimming(ARTrainset, lenXm, lenXl)
dailyPredInput = [lstmPth, arimaPth,dailyPredS, dailyPredE, testYs, testXs, testXm, testXl,  resultDf, ARTrainset]

resultDf30, y_test_pred, y_test, loss = ut.MonthlyPred(dateTerm, predS, lenXm ,lenXl, dailyPredInput )

print("Monthly Pred MSE loss : ",loss.item())

#resultDf : 28번의 예측값들을 모은 것
#y_test_pred : resultDf를 토대로 맨 마지막 값만 추가로 예측한 것
ut.PRPlot('MonthlyPredict', y_test_pred.detach().numpy(), y_test)


# PowerValue
termDay = pd.Timedelta(dateTerm, unit='days')  # 28일
lastDay = str(pd.Timestamp(predS) + termDay)  # 21-07-04

realPV = (resultDf.loc[predS : lastDay])[:-1]

resultDf = pd.DataFrame(resultDf, columns=["power_value"])
origin_max_pre_normalize = max(resultDf["power_value"])
origin_min_pre_normalize = min(resultDf["power_value"])

predPV = []
for i in y_test_pred.detach().numpy() :
    denorm = ut.denormalize(i, origin_max_pre_normalize, origin_min_pre_normalize)
    predPV.append(float(denorm[0]))
realPV["predict_pv"] = predPV
print(realPV)
#ut.PRPlot('Power_Value', realPV["predict_pv"], realPV["power_value"] )

