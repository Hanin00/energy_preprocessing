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
import Util as ut



pd.set_option('display.max_columns', None)
ARTrainset = pd.read_pickle("./data/ARTrainset.pkl")



# 데이터 불러오기(10분 단위 데이터)
df1D = pd.read_csv('./data/new_total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
dateColumn = 'updated'
freq = 'D'

resultDf = ut.slicerFreq(df1D, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간

lenXm = 7
lenXl = 28
testYs, testXs,  testXm, testXl = ut.testDataTrimming(ARTrainset, lenXm, lenXl)

testS = '2021-08-01'
testDate =  pd.Timedelta(28, unit='days')
startD = str(pd.Timestamp(testS) - testDate)  # 2021-07-04 00:00:00


#Daily Predict
arimaPth ="./model/arima_model_fit.pt"
lstmPth ="./model/LSTM_model.pt"

predS = '2021-07-04'
predE = '2021-08-01'


# resultDf - 일단위로 resample된 df(전체 데이터)

newDf, y_test_pred, y_test, loss =ut.DailyPredict(lstmPth, arimaPth,predS, predE, testYs, testXs, testXm, testXl,  resultDf, ARTrainset)

ut.PRPlot('DailyPredict', y_test_pred.detach().numpy(), y_test)
print("Daily Pred MSE loss : ",loss.item())