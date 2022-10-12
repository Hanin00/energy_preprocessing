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
import Util2 as ut2
import Util3 as ut3
import Util4 as ut4
import Util5 as ut5
import pickle



pd.set_option('display.max_columns', None)

# 데이터 불러오기(10분 단위 데이터)

df1D = pd.read_csv('./data/new_total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
dateColumn = 'updated'
freq = 'D'


resultDf = ut3.slicerFreq(df1D, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간

trainS = '2019-01-01'
trainE = '2021-07-31'
esPatience = 25
num_epochs = 200

'''
    Util : Xs에 LSTM 적용 X - ARIMA 예측값 X <- 수렴 X, 
    Util2 : Xs 에 LSTM 적용 O - ARIMA 예측값 X <- 수렴,  Traning MSE loss :  0.22228293120861053
    Util3 : Xs 만 LSTM 사용 - ARIMA 예측값 X <- 수렴 O, Xs - Traning MSE loss :  4.96087113788235e-06, Xm만 사용 - Traning MSE loss :  3.704776827362366e-05
    Util4 : Xs도 LSTM 사용 - ARIMA 예측값 X <- 수렴 X, 
    Util5 : Xs만 LSTM 사용 - ARIMA 예측값 O <- 수렴 O,Traning MSE loss :  0.00047351937973871827 <- ARIMA를 더했을 경우 더 안좋은 것을 알 수 있음
'''

# ARTrainset,y_train_pred ,y_train , loss = ut.Training(num_epochs, resultDf, trainS, trainE,esPatience)
# ARTrainset,y_train_pred ,y_train , loss = ut2.Training(num_epochs, resultDf, trainS, trainE,esPatience)
ARTrainset, y_train_pred ,y_train , loss = ut3.Training(num_epochs, resultDf, trainS, trainE)
# y_train_pred ,y_train , loss = ut4.Training(num_epochs, resultDf, trainS, trainE)
# ARTrainset, y_train_pred ,y_train , loss = ut5.Training(num_epochs, resultDf, trainS, trainE)
ut4.PRPlot('Training', y_train_pred.detach().numpy(), y_train.detach().numpy())
print("Traning MSE loss : ",loss.item())

# ARTrainset.to_pickle("./data/ARTrainset.pkl") #Util3,5



