# LSTM - https://eunhye-zz.tistory.com/entry/Pytorch%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-Timeseries-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%8D%B81-LSTM
# Arima Ex - https://wikidocs.net/50949
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
import tensorflow as tf
from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm


#1-1. LSTM 일단위 + FC, 데이터 형태 확인
    #1-1-1. FC 할 때 weight 를 한 건지..? 걍 결과값 O
#1-2. LSTM 주단위 + FC, 데이터 형태 확인
#1-3. LSTM 월단위 + FC, 데이터 형태 확인
#2-1. ARIMA 일단위 예측 + FC, 데이터 형태 확인
#3-1. 기존 방법에서 concat 방법 확인
#3-2. 기존 방법에서 Predict 한 방법 확인
#todo 4-1. 모듈화
#early stopping 코드 넣어야 함
#train한 모델 저장 및 불러와 test 할 수 있도록 코드 추가

#ARIMA predict reversr 후 lstm 결과값과 concat 하면 됨

# model 저장을 하고 predict를 하는지, 아니면 걍 값만 변경을 하는지, 값을 어떻게 주는지!




#graph 생성
def PRPlot(title,pred, real) :
    plt.title(title)
    plt.plot(pred, label="Preds", alpha=.7)
    plt.plot(real, label="Real")
    plt.legend()
    plt.show()

# 10T data -> 1D data resampling & 선형 보간
def slicerFreq(df, freq, dateColumn) :
    df.set_index(dateColumn, inplace=True)
    resultDf = df.resample(freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf["power_value"] = df_intp_linear[["power_value"]]
    return resultDf



# Sequence에 맞춰 데이터를 생성함.
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

#testS,E : test data를 나누는 구간
#train, test set으로 Df를 나눔. normalized를 함(MinMax Scaler)
# data ag -> test Xm, Xl 사이즈가 24, 3개로 너무 적기 때문에 test 데이터의 경우 Xs 길이까지 맞춰서

#testDate : xL의 크기
def datasetXsml(resultDf, seq_length, trainS, trainE, testS,testDate ):
    #todo 날짜로 인덱싱

    df = resultDf.copy()
    scaler = MinMaxScaler()
    df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))
    df = df["power_value"]
    df = df[::-1]

    train_max_pre_normalize, train_min_pre_normalize = 0.01,0.01
    test_max_pre_normalize, test_min_pre_normalize = 0.01,0.01
#[testS-28 : testS]

    if seq_length == 1 :

        train_set = df.loc[trainS:trainE]
        trainDf = pd.DataFrame(train_set, columns=["power_value"])

        train_max_pre_normalize = max(trainDf["power_value"])
        train_min_pre_normalize = min(trainDf["power_value"])

        trainDf["power_value"] = scaler.fit_transform(trainDf["power_value"].values.reshape(-1, 1))
        trainDf = trainDf["power_value"]
        trainX, trainY = build_dataset(np.array(trainDf), seq_length)

        startD = str(pd.Timestamp(testS) - testDate)  # 2021-07-04 00:00:00
        testYEndD = str(pd.Timestamp(testS) + testDate)  # 2021-08-29 00:00:00

        test_set = df.loc[startD: testS]  # testX 에 대한 Sequential Data 생성
        # test_set = df.loc[testS - testDate: testS] #testX 에 대한 Sequential Data 생성

        testDf = pd.DataFrame(test_set, columns=["power_value"])
        testDf["power_value"] = scaler.fit_transform(testDf["power_value"].values.reshape(-1, 1))
        #testDf = testDf["power_value"]


        test_max_pre_normalize = max(testDf["power_value"])
        test_min_pre_normalize = min(testDf["power_value"])

        testX, testY = build_dataset(np.array(testDf), seq_length)  # test Y는 8월 1일부터 8월 31일에 대한 값이 어야 함

        # print("testDf : ", testDf) 2021-07-04  - 2021-08-01
        # print("testS : ", testS) #2021-08-01
        # print("testYEndD : ", testYEndD) #2021-08-29 00:00:00
        testY = np.array(df.loc[testS: testYEndD][:-1])
    else :
        train_set = df.loc[trainS:trainE]
        trainDf = pd.DataFrame(train_set, columns=["power_value"])
        trainDf["power_value"] = scaler.fit_transform(trainDf["power_value"].values.reshape(-1, 1))
        trainDf = trainDf["power_value"]
        trainX, trainY = build_dataset(np.array(trainDf), seq_length)

        test_set = df.loc[testS:]
        testDf = pd.DataFrame(test_set, columns=["power_value"])
        testDf["power_value"] = scaler.fit_transform(testDf["power_value"].values.reshape(-1, 1))
        testDf = testDf["power_value"]
        testX, testY = build_dataset(np.array(testDf), seq_length)

    # 텐서로 변환
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)

    testX_tensor = torch.FloatTensor(testX) #size = [28,1]
    testY_tensor = torch.FloatTensor(testY) #size = [28]

    # 텐서 형태로 데이터 정의

    trainDataset = TensorDataset(trainX_tensor, trainY_tensor)
    testDataset = TensorDataset(testX_tensor, testY_tensor)


    # train_set = trainDf
    # test_set = testDf
    train_set = train_set[::-1]
    test_set = test_set[::-1]

    return train_set,test_set, trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, trainDataset, testDataset, train_max_pre_normalize ,train_min_pre_normalize,test_max_pre_normalize, test_min_pre_normalize

# #todo test dataset에 대해 denormalization 해서 값을 구한 후 denormalization 하게 되면 실제 값을 알 수 있음. 이 값을 predDataset에 추가해 다음 날의 데이터를 구해야함
# # todo Predict 값을 추가 해서 갱신하도록 코드 생성
# # newPV : 하루에 대해 예측 후 해당 값을 denomalize 해서 test Dataset에 추가함.
# def predDataset(newPV, testDataset ) :
#     testDataset += newPV


# Xm, Xl 데이터
class LSTMModel(nn.Module):
    def __init__(self,input_dim, inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim  # 128
        self.dropout_rate_ph = 0.02

        # base
        #self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim  # <-shape=(None, 12, 128)
        self.num_layers = num_layers  # 128
        self.hidden_dim = hidden_dim  # 128

        # layer
        self.activate_func = tf.nn.relu

        self.lstmXm = nn.LSTM(inputXm_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXl = nn.LSTM(inputXl_dim, hidden_dim, num_layers, batch_first=True)


    def forward(self, xs_input, xm_input, xl_input, hidden_dim, num_layers, output_dim):
        # todo xSinput는 xS의 input 수.. 아마..
        xSlstmRes = self.fcToLSTM_layer(xs_input, hidden_dim, num_layers, output_dim, 0)
        xMlstmRes = self.fcToLSTM_layer(xm_input, hidden_dim, num_layers, output_dim, 1) # LSTM은 h0, c0있지만 Fc 결과값은 out 만 있으니까!!!
        xLlstmRes = self.fcToLSTM_layer(xl_input, hidden_dim, num_layers, output_dim, 2)

        lenXl = xLlstmRes.size()[0]
        xSlstmRes = torch.flip(xSlstmRes,[0])  # tensor reverse
        xSlstmRes = xSlstmRes.split(lenXl, dim = 0)[0]

        xMlstmRes = torch.flip(xMlstmRes, [0])  # tensor reverse
        xMlstmRes = xMlstmRes.split(lenXl, dim=0)[0]


        concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)

        fc = nn.Linear(concatRes.size()[1], output_dim)
        fc1 = fc(concatRes[:, :])

        return fc1

    def fcToLSTM_layer(self, x, hidden_dim, num_layers, output_dim, whatIs):
        if whatIs == 0:  # short term data
            fc = nn.Linear(x.size()[1], output_dim)
            out = fc(x[:, :])

        else:  # long term data
            if whatIs == 1 :
                fc = nn.Linear(hidden_dim, output_dim)

                h0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
                c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()

                out, _ = self.lstmXm(x)

                out = fc(out[:, :])

            else :
                fc = nn.Linear(hidden_dim, output_dim)
                # # Initialize hidden state with zeros
                # h0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
                # # Initialize cell state
                # c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()

                out, _ = self.lstmXl(x)
                out = fc(out[:, :])

        return out


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

# Xs, Xm, Xl의 seq에 해당하는 Train Dataset 을 생성함 -> 결과값 중 ARIMA에 사용하는 Dataset은 Test 에서도 사용함
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

    # 텐서로 변환
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    # 텐서 형태로 데이터 정의
    trainDataset = TensorDataset(trainX_tensor, trainY_tensor)

    ARTrainset = train_set[::-1]

    return ARTrainset,trainX_tensor, trainY_tensor, trainDataset, train_max_pre_normalize ,train_min_pre_normalize


def Training(num_epochs, resultDf, trainS, trainE,esPatience ) :
    ARTrainset, trainXs_tensor, trainYs_tensor, trainDatasetXs, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 1, trainS, trainE)
    _, trainXm_tensor, trainYm_tensor, trainDatasetXm, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 7, trainS, trainE)
    _, trainXl_tensor, trainYl_tensor, trainDatasetXl, train_max_pre_normalize, train_min_pre_normalize = TrainDatasetCreater(resultDf, 28, trainS, trainE)

    input_dim = 1
    hidden_dim = 144
    num_layers = 2
    output_dim = 1
    inputXm_dim = trainXm_tensor.size()[1]
    inputXl_dim = trainXl_tensor.size()[1]

    # build model
    model = LSTMModel(input_dim, inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    hist = np.zeros(num_epochs)

    early_stop = EarlyStopping(esPatience)
    '''
        Auto Regressive - Arima model
        동일한 데이터(2019-01 ~ 2021-07)을 train 데이터로 사용
    '''
    arima_model = sm.tsa.arima.ARIMA(ARTrainset, order=(2, 1, 2), freq='D', missing="drop")
    arima_model_fit = arima_model.fit()
    # todo 원상 복구
    pred = arima_model_fit.get_prediction(start=pd.to_datetime('2019-01-01'), end=pd.to_datetime('2021-07-31'), )

    preds_arima = pred.predicted_mean
    preds_arima = torch.FloatTensor(preds_arima)
    preds_arima = torch.flip(preds_arima, [0])
    preds_arima = torch.unsqueeze(preds_arima, 1)


    print("LSTM start ")
    for t in range(num_epochs):
        y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
        # todo ARIMA랑 concat 후, Xs의 Y 값과 일치하는지 확인.
        # Y_train_pred 도 그러면 dimension 같아야겠지 ~
        y_train = torch.flip(trainYs_tensor, [0])  # tensor reverse
        y_train = y_train.split(len(y_train_pred), dim=0)[0]
        preds_arima = preds_arima.split(len(y_train_pred), dim=0)[0]
        y_train_pred += preds_arima
        y_train_pred = torch.flip(y_train_pred, [0])
        loss = loss_fn(y_train_pred, y_train)
        # early stopping
        early_stop.step(loss.item())
        if early_stop.is_stop():
            break
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())

        hist[t] = loss.item()
        # Zero out  "gradient, else t hey will accumulate between epochs
        optimiser.zero_grad()
        # Backward pass
        loss.backward()
        # Update parameters
        optimiser.step()

    # todo model 저장
    torch.save(model, './model/LSTM_model.pt')
    torch.save(arima_model_fit, './model/arima_model_fit.pt')

    PRPlot('Train1- origin', y_train_pred.detach().numpy(), y_train)
    PRPlot('Train1- add loss', y_train_pred.detach().numpy() + loss.item(), y_train)

    return ARTrainset









pd.set_option('display.max_columns', None)

# 데이터 불러오기(10분 단위 데이터)

df1D = pd.read_csv('../data/total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
dateColumn = 'updated'
freq = 'D'

resultDf = slicerFreq(df1D, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간



trainS = '2019-01-01'
trainE = '2021-07-31'
esPatience = 20
num_epochs = 20

ARTrainset = Training(num_epochs, resultDf, trainS, trainE,esPatience)

ARTrainset,trainXs_tensor, trainYs_tensor, trainDatasetXs, train_max_pre_normalize ,train_min_pre_normalize = TrainDatasetCreater(resultDf, 1, trainS, trainE )
_,trainXm_tensor,    trainYm_tensor, trainDatasetXm, train_max_pre_normalize ,train_min_pre_normalize = TrainDatasetCreater(resultDf, 7, trainS, trainE )
_,trainXl_tensor,   trainYl_tensor, trainDatasetXl, train_max_pre_normalize ,train_min_pre_normalize = TrainDatasetCreater(resultDf, 28, trainS, trainE )


# print(train_max_pre_normalize)
# print(train_min_pre_normalize)


# Build model


input_dim = 1
hidden_dim = 144
num_layers = 2
output_dim = 1
inputXm_dim = trainXm_tensor.size()[1]
inputXl_dim = trainXl_tensor.size()[1]

#build model
model = LSTMModel(inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim)
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


# Train model
num_epochs = 20
hist = np.zeros(num_epochs)


#early_stop = EarlyStopping(patience=30)
'''
    Auto Regressive - Arima model
    동일한 데이터(2019-01 ~ 2021-07)을 train 데이터로 사용
'''
arima_model  = sm.tsa.arima.ARIMA(ARTrainset, order = (2,1,2),freq = 'D',missing = "drop")
arima_model_fit = arima_model.fit()
#todo 원상 복구
pred = arima_model_fit.get_prediction(start=pd.to_datetime('2019-01-01'), end = pd.to_datetime('2021-07-31'), )

preds_arima = pred.predicted_mean
preds_arima = torch.FloatTensor(preds_arima)
preds_arima = torch.flip(preds_arima, [0])
preds_arima = torch.unsqueeze(preds_arima, 1)

print("LSTM start " )
for t in range(num_epochs):
    y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
    #todo ARIMA랑 concat 후, Xs의 Y 값과 일치하는지 확인.
    # Y_train_pred 도 그러면 dimension 같아야겠지 ~
    y_train = torch.flip(trainYs_tensor, [0])  # tensor reverse
    y_train = y_train.split(len(y_train_pred), dim=0)[0]
    preds_arima = preds_arima.split(len(y_train_pred), dim=0)[0]
    y_train_pred += preds_arima
    y_train_pred = torch.flip(y_train_pred, [0])
    loss = loss_fn(y_train_pred, y_train)
#early stopping
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

#todo model 저장
torch.save(model, './model/LSTM_model.pt')
torch.save(arima_model_fit, './model/arima_model_fit.pt')

## Train Fitting
plt.title('Train1 - origin')
plt.plot(y_train_pred.detach().numpy(), label="Preds", alpha = .7)
plt.plot(y_train.detach().numpy(), label="Real", alpha = .7)
#plt.xticks(ARTrainset.index)
plt.legend()
plt.show()

## Train Fitting
plt.title('Train2')
#plt.plot(y_train_pred.detach().numpy(), label="Preds", alpha = .7)
plt.plot(y_train_pred.detach().numpy()+loss.item(), label="Preds", alpha = .7)
plt.plot(y_train.detach().numpy(), label="Real", alpha = .7)
#plt.xticks(ARTrainset.index)
plt.legend()
plt.show()




#Predict
def denormalize(y,max_pre_normalize ,min_pre_normalize ):
    final_value = y*(max_pre_normalize - min_pre_normalize) + min_pre_normalize
    return final_value

# todo read me update -> first test Dataset
'''
    beforeDf : 예측 시점 이전의 데이터
    lenXl, lenXm :  
'''
def testDataTrimming(beforeDf, lenXm,lenXl) :
    testYs = np.array(beforeDf[::-1]) # 큰 값부터(최근 값 부터)
    testYs = testYs[:lenXl]
    testXs = torch.FloatTensor(testYs[1:lenXl+2])

    testXm,_ = build_dataset(np.array(beforeDf[:: -1]), lenXm)
    testXl,_= build_dataset(np.array(beforeDf[::-1]), lenXl)

    testXm = testXm[:lenXl] # 제일 긴 길이인 lenXl에 맞춰서 데이터를 잘라즘. 이 데이터는 현재 최신 데이터부터 예전 데이터 순서로 졍렬되어 있음
    testXl = testXl[:lenXl]

    testXs = torch.FloatTensor(np.array(testXs[::-1]))
    testYs = torch.FloatTensor(np.array(testYs[::-1]))
    testXm = torch.FloatTensor(np.array(testXm[::-1]))
    testXl = torch.FloatTensor(np.array(testXl[::-1]))


    testXs = torch.flip(testXs, [-1])
    testXs = testXs.unsqueeze(-1)
    testYs = torch.flip(testYs, [-1])
    testXm = torch.flip(testXm, [-1])
    testXl = torch.flip(testXl, [-1])

    return testYs, testXs, testXm, testXl


lenXm = 7
lenXl = 28
testYs, testXs,  testXm, testXl = testDataTrimming(ARTrainset, lenXm, lenXl)

print("size")
print(testYs.size())
print(testXs.size())
print(testXm.size())
print(testXl.size())
sys.exit()






testS = '2021-08-01'
testDate =  pd.Timedelta(28, unit='days')
startD = str(pd.Timestamp(testS) - testDate)  # 2021-07-04 00:00:00
#testDate =  28

# make predictions
'''
    최근 한달(2021-08)월에 대해 예측
    08월은 Y값으로 주어짐
'''
'''
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
    1. 학습된 모델을 불러옴
    2. 하루 예측 후 새로운 데이터로 갱신함
    3. 1,2 번을 30번 반복
'''

def PRPlot(title,pred, real) :
    plt.title(title)
    plt.plot(pred+ loss.item(), label="Preds", alpha=.7)
    plt.plot(real, label="Real")
    plt.legend()
    plt.show()

#일단 하루 예측 먼저 -> 30번 반복
def DailyPredict(lstmPth, arimaPth,predS, predE,testXs_tensor, testXm_tensor, testXl_tensor, hidden_dim, num_layers, output_dim, resultDf ) :
    model = torch.load(lstmPth)
    arima_model_fit = torch.load(arimaPth)

    testXs_tensor = torch.squeeze(testXs_tensor,1)

    y_test_pred = model(testXs_tensor, testXm_tensor, testXl_tensor, hidden_dim, num_layers, output_dim)

    y_test = torch.flip(testYs_tensor, [0])  # tensor reverse
    y_test = y_test.split(len(y_test_pred), dim=0)[0]

    # todo 최근 한 달에 대한 arima 예측
    arima_test_pred = arima_model_fit.get_prediction(start=pd.to_datetime(predS), end=pd.to_datetime(predE), )
    arima_test_pred = arima_test_pred.predicted_mean  # : 31
    arima_test_pred = torch.FloatTensor(arima_test_pred)
    arima_test_pred = torch.flip(arima_test_pred, [0])
    arima_test_pred = torch.unsqueeze(arima_test_pred, 1)
    arima_test_pred = arima_test_pred.split(len(y_test_pred), dim=0)[0]

    y_test_pred += arima_test_pred
    # y_test_pred = torch.flip(y_test_pred, [0])

    loss = loss_fn(y_test_pred, y_test)

    # todo predict 값
    predLast = y_test_pred.detach().numpy()[-1]  # train Dataset의 normalize 된 마지막 값
    predLast -= 0.000001  # float이라 정확한 값 비교가 어려워서 부등호를 이용함. 이때 사용하기 위해 값을 임의로 줄임
    testset = pd.DataFrame(Testset, columns=["power_value"])
    testFirDenorm = denormalize(predLast, test_max_pre_normalize, test_min_pre_normalize)  # denormalize 된 원래 의 값

    origin_max_pre_normalize = max(resultDf["power_value"])
    origin_min_pre_normalize = min(resultDf["power_value"])
    testDenorm = float(denormalize(testFirDenorm, origin_max_pre_normalize, origin_min_pre_normalize))  # 마지막 날짜에 대한 예측
    # testDenorm = 새로 예측한 수요량

    exdf = resultDf[trainS: trainE]
    newPv = testDenorm
    newDf = appendNewData(exdf, 28, newPv)

    PRPlot('Test1- origin', y_test_pred.detach().numpy(), y_test)
    PRPlot('Test1- add loss', y_test_pred.detach().numpy() + loss.item(), y_test)

    return newDf

#Daily Predict
arimaPth ='./model/arima_model_fit.pt'
lstmPth ='./model/model.pt'

predS = '2021-07-04'
predE = '2021-08-01'

# resultDf - 일단위로 resample된 df(전체 데이터)

newDf = DailyPredict(lstmPth, arimaPth,predS, predE,testXs, testXm, testXl, hidden_dim, num_layers, output_dim, resultDf  )
print(newDf.head())


sys.exit()


predS = '2021-08-01'
def MonthlyPred(predS, testDate) :
    testDate = pd.Timedelta(testDate, unit='days')
    test





for i in range(30) :
    # Daily Predict
    arimaPth = './model/arima_model_fit.pt'
    lstmPth = './model/model.pt'

    print(testXs_tensor)
    print(type(testXs_tensor))
    sys.exit()

    # resultDf - 일단위로 resample된 df(전체 데이터)
    newDf = DailyPredict(lstmPth, arimaPth, predS, predE, testXs_tesnsor, testXm_tensor, testXl_tensor, hidden_dim,
                         num_layers, output_dim, resultDf)





sys.exit()

''' 
    학습과 테스트 이어서 예측..
    -> 테스트 결과값이 한 달에 대한 예측같지가 않음
    예측 후 
'''

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

plt.title('graph3')
plt.plot(y_test_pred, label="Preds")
plt.plot(y_test, label="Real")
plt.legend()
plt.show()

test_seq = x_test[:1]  ## X_test에 있는 데이터중 첫번째것을 가지고 옮
preds = []

for _ in range(len(x_test)):
    # model.init_hidden(test_seq)
    y_test_pred = model(test_seq)

    pred = y_test_pred.item()
    preds.append(pred)
    new_seq = test_seq.numpy()
    new_seq = np.append(new_seq, pred)
    new_seq = new_seq[1:]  ## index가 0인 것은 제거하여 예측값을 포함하여 7일치 데이터 구성
    test_seq = torch.from_numpy(new_seq).view(1, 6, 1).float()

plt.title('graph4')
plt.plot(preds, label="Preds")
plt.plot(y_test, label="Data")
# plt.plot(y_test.detach().numpy(), label="Data")
plt.legend()
plt.show()
