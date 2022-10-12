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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import tensorflow as tf
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


# Xs에 LSTM 적용

class LSTMModel(nn.Module):
    def __init__(self,input_dim, inputXs_dim, inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim  # 128
        #self.dropout_rate_ph = 0.02

        # base
        self.input_dim = input_dim  # <-shape=(None, 12, 128)
        self.num_layers = num_layers  # 128
        self.hidden_dim = hidden_dim  # 128

        # layer
        #self.activate_func = tf.nn.relu

        # self.fc = nn.Linear(hidden_dim, output_dim)

        self.fcXs = nn.Linear(hidden_dim, output_dim)
        self.fcXm = nn.Linear(hidden_dim, output_dim)
        self.fcXl = nn.Linear(hidden_dim, output_dim)


        self.lstmXs = nn.LSTM(inputXs_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXm = nn.LSTM(inputXm_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXl = nn.LSTM(inputXl_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, xs_input, xm_input, xl_input, hidden_dim, num_layers, output_dim):
        #xSlstm
        h0Xs = torch.zeros(self.num_layers, xs_input.size(0), self.hidden_dim).requires_grad_()  # 2, 942, 128
        c0Xs = torch.zeros(self.num_layers, xs_input.size(0), self.hidden_dim).requires_grad_()  # 2, 942, 128

        # out, _ = self.lstmXs(x)
        xSlstmRes, (hn, cn) = self.lstmXs(xs_input)
        # xSlstmRes, (hn, cn) = self.lstmXs(xs_input, (h0Xs.detach(), c0Xs.detach()))
        xSlstmRes = self.fcXs(xSlstmRes[:, :])


        #xMlstm
        h0Xm = torch.zeros(self.num_layers, xm_input.size(0), self.hidden_dim).requires_grad_()
        c0Xm = torch.zeros(self.num_layers, xm_input.size(0), self.hidden_dim).requires_grad_()

        # out, _ = self.lstmXm(x)
        xMlstmRes,(hn, cn) = self.lstmXm(xm_input)
        # xMlstmRes, (hn, cn) = self.lstmXm(xm_input, (h0Xm.detach(), c0Xm.detach()))

        xMlstmRes = self.fcXm(xMlstmRes[:,  :])


        #xLlstm
        h0Xl = torch.zeros(self.num_layers, xl_input.size(0), self.hidden_dim).requires_grad_()
        c0Xl = torch.zeros(self.num_layers, xl_input.size(0), self.hidden_dim).requires_grad_()

        # out, _ = self.lstmXl(x)
        xLlstmRes,(hn, cn) = self.lstmXl(xl_input)
        # xLlstmRes, (hn, cn) = self.lstmXl(xl_input, (h0Xl.detach(), c0Xl.detach()))
        xLlstmRes = self.fcXl(xLlstmRes[:, :])

       #Concat
        lenXl = xLlstmRes.size()[0]
        # xSlstmRes = torch.flip(xSlstmRes, [0])  # tensor reverse
        # xSlstmRes = xSlstmRes.split(lenXl, dim=0)[0]

        xMlstmRes = torch.flip(xMlstmRes, [0])  # tensor reverse
        xMlstmRes = xMlstmRes.split(lenXl, dim=0)[0]

        concatRes = torch.cat(( xMlstmRes, xLlstmRes), self.input_dim)
        # concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)

        self.fc = nn.Linear(concatRes.size()[1], output_dim)
        fc1 = self.fc(concatRes[:, :])

        return fc1


#graph 생성
def PRPlot(title,pred, real) :
    plt.title(title)
    plt.plot(pred, label="Preds", alpha=.7)
    plt.plot(real, label="Real")
    plt.legend()
    plt.show()
    plt.savefig("./output/"+title+".png")

# 10T data -> 1D data resampling & 선형 보간
def slicerFreq(df, freq, dateColumn) :
    df.set_index(dateColumn, inplace=True)
    resultDf = df.resample(freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf["power_value"] = df_intp_linear[["power_value"]]
    return resultDf

# Sequence에 맞춰 데이터를 생성
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length]
        _y = time_series[i + seq_length]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience

    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1

    def is_stop(self):
        return self.patience >= self.patience_limit

# Xs, Xm, Xl의 seq에 해당하는 Train Dataset 을 생성함 -> 결과값 중 ARIMA에 사용하는 Dataset은 Test 에서 사용
def TrainDatasetCreater(resultDf, seq_length, trainS, trainE ):
    df = resultDf.copy()
    scaler = MinMaxScaler()
    df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))
    df = df["power_value"]
    df = df[::-1]
    train_max_pre_normalize, train_min_pre_normalize = 0.01,0.01

    if seq_length == 1 :

        train_set = df.loc[trainS:trainE]
        trainDf = pd.DataFrame(train_set, columns=["power_value"])

        train_max_pre_normalize = max(trainDf["power_value"])
        train_min_pre_normalize = min(trainDf["power_value"])

        trainDf["power_value"] = scaler.fit_transform(trainDf["power_value"].values.reshape(-1, 1))
        trainDf = trainDf["power_value"]
        trainX, trainY = build_dataset(np.array(trainDf), seq_length)

    else :
        train_set = df.loc[trainS:trainE]
        trainDf = pd.DataFrame(train_set, columns=["power_value"])
        trainDf["power_value"] = scaler.fit_transform(trainDf["power_value"].values.reshape(-1, 1))
        trainDf = trainDf["power_value"]
        trainX, trainY = build_dataset(np.array(trainDf), seq_length)


    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)

    trainDataset = TensorDataset(trainX_tensor, trainY_tensor)

    ARTrainset = train_set[::-1]

    return ARTrainset,trainX_tensor, trainY_tensor, trainDataset, train_max_pre_normalize ,train_min_pre_normalize


def Training(num_epochs, resultDf, trainS, trainE,esPatience ) :
    ARTrainset, trainXs_tensor, trainYs_tensor, trainDatasetXs, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 2, trainS, trainE)
    _, trainXm_tensor, trainYm_tensor, trainDatasetXm, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 7, trainS, trainE)
    _, trainXl_tensor, trainYl_tensor, trainDatasetXl, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 28, trainS, trainE)

    input_dim = 1
    hidden_dim = 128
    # hidden_dim = 144
    num_layers = 2
    output_dim = 1

    inputXs_dim = trainXs_tensor.size()[1]
    inputXm_dim = trainXm_tensor.size()[1]
    inputXl_dim = trainXl_tensor.size()[1]

    # build model
    model = LSTMModel(input_dim, inputXs_dim, inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    hist = np.zeros(num_epochs)
    # early_stop = EarlyStopping(esPatience)
    arima_model = sm.tsa.arima.ARIMA(ARTrainset, order=(2, 1, 2), freq='D', missing="drop")
    arima_model_fit = arima_model.fit()
    pred = arima_model_fit.get_prediction(start=pd.to_datetime('2019-01-01'), end=pd.to_datetime('2021-07-31'), )

    preds_arima = pred.predicted_mean
    preds_arima = torch.FloatTensor(preds_arima)
    preds_arima = torch.flip(preds_arima, [0])
    preds_arima = torch.unsqueeze(preds_arima, 1)

    print("LSTM start ")
    for t in range(num_epochs):
        y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
        y_train = torch.flip(trainYs_tensor, [0])  # tensor reverse
        # y_train = y_train.split(len(y_train_pred), dim=0)[0]
        y_train = y_train.split(len(y_train_pred), dim=0)[0]
        # preds_arima = preds_arima.split(len(y_train_pred), dim=0)[0]
        # y_train_pred += preds_arima
        # y_train_pred = torch.flip(y_train_pred, [0])
        loss = loss_fn(y_train_pred, y_train)

        # early stopping
        # early_stop.step(loss.item())
        # if early_stop.is_stop():
        #     break
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())

        hist[t] = loss.item()
        # Zero out  "gradient, else t hey will accumulate between epochs
        optimiser.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        optimiser.step()

    torch.save(model, './model/LSTM_model.pt')
    torch.save(arima_model_fit, './model/arima_model_fit.pt')

    # PRPlot('Train1- origin', y_train_pred.detach().numpy(), y_train)
    # PRPlot('Train1- add loss', y_train_pred.detach().numpy() + loss.item(), y_train)
    return ARTrainset,y_train_pred ,y_train , loss

#Predict
def denormalize(y,max_pre_normalize ,min_pre_normalize ):
    final_value = y*(max_pre_normalize - min_pre_normalize) + min_pre_normalize
    return final_value

'''
    beforeDf : 예측 시점 이전의 데이터
'''
def testDataTrimming(beforeDf, lenXm, lenXl) :
    testYs = np.array(beforeDf[::-1]) # 큰 값부터(최근 값 부터)
    testYs = testYs[:lenXl]


    testXs = np.array(beforeDf[::-1])
    testXs = testXs[1:lenXl+1]
    testXs = torch.FloatTensor(testXs)


    testXm,_ = build_dataset(np.array(beforeDf[:: -1]), lenXm)
    testXl,_= build_dataset(np.array(beforeDf[::-1]), lenXl)

    testXm = testXm[:lenXl] # 제일 긴 길이인 lenXl에 맞춰서 데이터를 잘라즘. 이 데이터는 현재 최신 데이터부터 예전 데이터 순서로 졍렬되어 있음
    testXl = testXl[:lenXl]

    testYs = torch.FloatTensor(np.array(testYs))
    testXs = torch.FloatTensor(np.array(testXs))
    testXm = torch.FloatTensor(np.array(testXm))
    testXl = torch.FloatTensor(np.array(testXl))


    testXs = torch.flip(testXs, [-1])
    testXs = testXs.unsqueeze(-1)
    testYs = torch.flip(testYs, [-1])
    testXm = torch.flip(testXm, [-1])
    testXl = torch.flip(testXl, [-1])

    return testYs, testXs, testXm, testXl

# make predictions
'''
    최근 한달(2021-08)월에 대해 예측
    08월은 Y값으로 주어짐

    originTSetDf : Origin Test Set, Dataframe
    datalen : XL의 데이터 개수    
    newPv : 바로 앞에서 예측한 Power_value의 Denormalize 값
'''
def appendNewData(originTSetDt, dataLen, newPv) :
    td_pd = pd.Timedelta(1, unit='days')
    nextIdx = pd.Timestamp(pd.Timestamp(pd.to_datetime(originTSetDt.index[-1].date()) + td_pd).date())
    originTSetDt.loc[str(nextIdx)] = newPv
    testXDf = originTSetDt[::-1]
    testXDf = testXDf[:dataLen]
    return testXDf
''' 
    한 달 예측
    1. 학습된 모델을 불러옴
    2. 하루 예측 후 새로운 데이터로 갱신함
    3. 1,2 번을 30번 반복
'''

def DailyPredict(lstmPth, arimaPth,predS, predE,testYs_tensor , testXs_tensor, testXm_tensor, testXl_tensor, resultDf, ARTrainset ) :
    model = torch.load(lstmPth)
    arima_model_fit = torch.load(arimaPth)
    #testXs_tensor = torch.squeeze(testXs_tensor,1)

    input_dim = 1
    hidden_dim = 144
    num_layers = 2
    output_dim = 1
    loss_fn = torch.nn.MSELoss()

    y_test_pred = model(testXs_tensor, testXm_tensor, testXl_tensor, hidden_dim, num_layers, output_dim)

    y_test = torch.flip(testYs_tensor, [0])  # tensor reverse
    y_test = y_test.split(len(y_test_pred), dim=0)[0]
    y_test = torch.flip(y_test, [0])  # tensor reverse

    arima_test_pred = arima_model_fit.get_prediction(start=pd.to_datetime(predS), end=pd.to_datetime(predE), )
    arima_test_pred = arima_test_pred.predicted_mean  # : 31
    arima_test_pred = torch.FloatTensor(arima_test_pred)
    arima_test_pred = torch.flip(arima_test_pred, [0])
    arima_test_pred = torch.unsqueeze(arima_test_pred, 1)
    arima_test_pred = arima_test_pred.split(len(y_test_pred), dim=0)[0]

    y_test_pred += arima_test_pred
    y_test_pred = torch.flip(y_test_pred, [0])

    loss = loss_fn(y_test_pred, y_test)

#denormalize
    predLast = y_test_pred.detach().numpy()[-1]  # train Dataset의 normalize 된 마지막 값
    predLast -= 0.000001  # float이라 정확한 값 비교가 어려워서 부등호를 이용함. 이때 사용하기 위해 값을 임의로 줄임
    testset = pd.DataFrame(ARTrainset, columns=["power_value"])
    test_max_pre_normalize = max(testset["power_value"])
    test_min_pre_normalize = min(testset["power_value"])
    testFirDenorm = denormalize(predLast, test_max_pre_normalize, test_min_pre_normalize)  # denormalize 된 원래 의 값


    resultDf = pd.DataFrame(resultDf, columns=["power_value"])
    origin_max_pre_normalize = max(resultDf["power_value"])
    origin_min_pre_normalize = min(resultDf["power_value"])
    testDenorm = float(denormalize(testFirDenorm, origin_max_pre_normalize, origin_min_pre_normalize))  # 마지막 날짜에 대한 예측

    exdf = ARTrainset[predS: predE]
    newPv = testDenorm
    newDf = appendNewData(exdf, 28, newPv)

    return newDf, y_test_pred, y_test,  loss



def MonthlyPred(dateTerm, predS, lenXm ,lenXl, dailyPredInput ) :
    nDay = pd.Timedelta(1, unit='days')
    termDay = pd.Timedelta(dateTerm, unit='days')  # 28일

    lstmPth, arimaPth, predS_Daily, predE, testYs, testXs, testXm, testXl, resultDf, ARTrainset = dailyPredInput

    testYs, testXs, testXm, testXl = testDataTrimming(ARTrainset, lenXm, lenXl)

    # Daily Predict
    # arimaPth = './model/arima_model_fit.pt'
    # lstmPth = './model/model.pt'
    predE = predS  # 21-08-01
    for i in range(dateTerm):
        predS_Daily = str(pd.Timestamp(predE) - termDay)  # 21-07-04
        newDf, y_test_pred, y_test, loss = DailyPredict(lstmPth, arimaPth, predS_Daily, predE, testYs, testXs, testXm, testXl, resultDf, ARTrainset)
        ARTrainset = newDf
        predE = str(pd.Timestamp(predE) + nDay)
        resultDf = ARTrainset

    return resultDf, y_test_pred, y_test, loss


