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

# Xs, Xm, Xl 에 LSTM 적용
# 학습 시에는 1일 단위로 잘 예측하는지 확인하고, 예측 시에는 매일 하루씩 예측한 값을 더해 올림
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim,xl_len):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstmXs = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXl = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        # self.fc = nn.Linear( xl_len, self.output_dim)  #756x1
        self.fcXs = nn.Linear(self.hidden_dim, self.output_dim)
        self.fcXm = nn.Linear(self.hidden_dim, self.output_dim)
        self.fcXl = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, xs,xm,xl):        # Initialize hidden state with
        #Xs
        hs0 = torch.zeros(self.num_layers, xs.size(0), self.hidden_dim).requires_grad_()
        cs0 = torch.zeros(self.num_layers, xs.size(0), self.hidden_dim).requires_grad_()
        outXs, (hn, cn) = self.lstmXs(xs, (hs0.detach(), cs0.detach()))
        outXs = self.fcXs(outXs[:, -1, :])
        #Xm
        hm0 = torch.zeros(self.num_layers, xm.size(0), self.hidden_dim).requires_grad_()
        cm0 = torch.zeros(self.num_layers, xm.size(0), self.hidden_dim).requires_grad_()
        outXm, (hn, cn) = self.lstmXm(xm, (hm0.detach(), cm0.detach()))
        outXm = self.fcXm(outXm[:, -1, :])

        #Xl
        hl0 = torch.zeros(self.num_layers, xl.size(0), self.hidden_dim).requires_grad_()
        cl0 = torch.zeros(self.num_layers, xl.size(0), self.hidden_dim).requires_grad_()
        outXl, (hn, cn) = self.lstmXl(xl, (hl0.detach(), cl0.detach()))
        outXl = self.fcXl(outXl[:, -1, :])

        # Concat
        lenXl = outXl.size()[0]
        xSlstmRes = torch.flip(outXs, [0])  # tensor reverse #유효한 데이터만 사용 - 최근 날짜부터 outXl 개수에 맞춰서 자름
        xSlstmRes = xSlstmRes.split(lenXl, dim=0)[0]
        xSlstmRes = torch.flip(xSlstmRes, [0])  # tensor reverse

        xMlstmRes = torch.flip(outXm, [0])  # tensor reverse
        xMlstmRes = xMlstmRes.split(lenXl, dim=0)[0]
        xMlstmRes = torch.flip(xMlstmRes, [0])  # tensor revers'e

        # concatRes = torch.cat((xSlstmRes,xMlstmRes, outXl), self.input_dim ) #756x3
        concatRes = torch.cat((xSlstmRes,xMlstmRes, outXl), self.input_dim)
        # concatRes = torch.sum(concatRes, self.input_dim)
        concatRes = torch.mean(concatRes, self.input_dim)  # 세 예측값의 평균 사용함
        # print(concatRes.size())

        return concatRes



def Training(num_epochs, resultDf, trainS, trainE ) :
    ARTrainset, trainXs_tensor,  xs_test, trainYs_tensor, ys_test= TrainDatasetCreater(resultDf, 2, trainS, trainE)
    _,trainXm_tensor, xm_test, trainYm_tensor, ym_test= TrainDatasetCreater(resultDf, 8, trainS, trainE)
    _,trainXl_tensor, xl_test, trainYl_tensor, yl_test= TrainDatasetCreater(resultDf, 29, trainS, trainE)




    input_dim = 1
    hidden_dim = 128
    num_layers = 2
    output_dim = 1

    inputXs_dim = trainXs_tensor.size()[1]
    inputXm_dim = trainXm_tensor.size()[1]
    inputXl_dim = trainXl_tensor.size()[1]


    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,len(trainXl_tensor))
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    hist = np.zeros(num_epochs)


    print("LSTM start ")
    for t in range(num_epochs):
        y_train_pred = model(trainXs_tensor,trainXm_tensor,trainXl_tensor)

        y_train = torch.flip(trainYs_tensor, [0])  #Xl 크기만큼만 유효하므로 뒤집어서 자르고 다시 뒤집음. 
        # 학습 시에는 일단위 예측을 정확하게 하는 것이 목표
        y_train = y_train.split(len(trainYl_tensor), dim=0)[0]
        y_train = torch.flip(y_train, [0])

        print("y_train_pred[0] : ", y_train_pred[0], "y_train[0] : ",y_train[0])

        loss = loss_fn(y_train_pred, y_train)

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

    return y_train_pred ,y_train , loss



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

def load_data(stock, look_back):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));  # 220
    train_set_size = data.shape[0] - (test_set_size);  # 881

    x_train = data[:train_set_size, :-1, :]  # (881,6,1) 이전 7일까지의 값을 사용하니까
    y_train = data[:train_set_size, -1, :]  # (881,1)

    x_test = data[train_set_size:, :-1]  # (220,1)
    y_test = data[train_set_size:, -1, :]  # (881,1)

    return [x_train, y_train, x_test, y_test]

# Xs, Xm, Xl의 seq에 해당하는 Train Dataset 을 생성함 -> 결과값 중 ARIMA에 사용하는 Dataset은 Test 에서 사용
def TrainDatasetCreater(resultDf, seq_length, trainS, trainE ):
    df = resultDf.copy()
    scaler = MinMaxScaler()
    df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))

    x_train, y_train, x_test, y_test = load_data(df, seq_length)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    train_set = df.loc[trainS:trainE] # normalize 된 값


    trainDf = pd.DataFrame(train_set, columns=["power_value"])
    ARTrainset = train_set[::-1]

    return ARTrainset,x_train, x_test, y_train, y_test


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

# def DailyPredict(lstmPth, arimaPth,predS, predE,testYs_tensor , testXs_tensor, testXm_tensor, testXl_tensor, resultDf, ARTrainset ) :
def DailyPredict(lstmPth, predS, predE,testYs_tensor , testXs_tensor, testXm_tensor, testXl_tensor, resultDf ) :
    model = torch.load(lstmPth)
    # arima_model_fit = torch.load(arimaPth)
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

    # arima_test_pred = arima_model_fit.get_prediction(start=pd.to_datetime(predS), end=pd.to_datetime(predE), )
    # arima_test_pred = arima_test_pred.predicted_mean  # : 31
    # arima_test_pred = torch.FloatTensor(arima_test_pred)
    # arima_test_pred = torch.flip(arima_test_pred, [0])
    # arima_test_pred = torch.unsqueeze(arima_test_pred, 1)
    # arima_test_pred = arima_test_pred.split(len(y_test_pred), dim=0)[0]

    y_test_pred += arima_test_pred
    y_test_pred = torch.flip(y_test_pred, [0])

    loss = loss_fn(y_test_pred, y_test)

#denormalize
    predLast = y_test_pred.detach().numpy()[-1]  # train Dataset의 normalize 된 마지막 값
    predLast -= 0.000001  # float이라 정확한 값 비교가 어려워서 부등호를 이용함. 이때 사용하기 위해 값을 임의로 줄임
    # testset = pd.DataFrame(ARTrainset, columns=["power_value"])
    test_max_pre_normalize = max(testset["power_value"])
    test_min_pre_normalize = min(testset["power_value"])
    testFirDenorm = denormalize(predLast, test_max_pre_normalize, test_min_pre_normalize)  # denormalize 된 원래 의 값

    resultDf = pd.DataFrame(resultDf, columns=["power_value"])
    origin_max_pre_normalize = max(resultDf["power_value"])
    origin_min_pre_normalize = min(resultDf["power_value"])
    testDenorm = float(denormalize(testFirDenorm, origin_max_pre_normalize, origin_min_pre_normalize))  # 마지막 날짜에 대한 예측

    # exdf = ARTrainset[predS: predE]
    newPv = testDenorm
    # newDf = appendNewData(exdf, 28, newPv)

    return newDf, y_test_pred, y_test,  loss



def MonthlyPred(dateTerm, predS, lenXm ,lenXl, dailyPredInput ) :
    nDay = pd.Timedelta(1, unit='days')
    termDay = pd.Timedelta(dateTerm, unit='days')  # 28일

    # lstmPth, arimaPth, predS_Daily, predE, testYs, testXs, testXm, testXl, resultDf, ARTrainset = dailyPredInput
    lstmPth,  predS_Daily, predE, testYs, testXs, testXm, testXl, resultDf = dailyPredInput

    # testYs, testXs, testXm, testXl = testDataTrimming(ARTrainset, lenXm, lenXl)
    testYs, testXs, testXm, testXl = testDataTrimming(ARTrainset, lenXm, lenXl)

    # Daily Predict
    # arimaPth = './model/arima_model_fit.pt'
    # lstmPth = './model/model.pt'
    predE = predS  # 21-08-01
    for i in range(dateTerm):
        predS_Daily = str(pd.Timestamp(predE) - termDay)  # 21-07-04
        newDf, y_test_pred, y_test, loss = DailyPredict(lstmPth, predS_Daily, predE, testYs, testXs, testXm, testXl, resultDf, ARTrainset)
        ARTrainset = newDf
        predE = str(pd.Timestamp(predE) + nDay)
        resultDf = ARTrainset

    return resultDf, y_test_pred, y_test, loss


