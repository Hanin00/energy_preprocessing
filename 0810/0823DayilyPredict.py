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
import statsmodels.api as sm


#todo 1-1. LSTM 일단위 + FC, 데이터 형태 확인
    #todo 1-1-1. FC 할 때 weight 를 한 건지..? 걍 결과값?
#todo 1-2. LSTM 주단위 + FC, 데이터 형태 확인
#todo 1-3. LSTM 월단위 + FC, 데이터 형태 확인
#todo 2-1. ARIMA 일단위 예측 + FC, 데이터 형태 확인
#todo 3-1. 기존 방법에서 concat 방법 확인
#todo 3-2. 기존 방법에서 Predict 한 방법 확인
#todo 4-1. 모듈화


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

#train, test set으로 Df를 나눔. normalized를 함(MinMax Scaler)
def datasetXsml(df, seq_length ):
    scaler = MinMaxScaler()
    df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))
    df = df["power_value"]
    df = df[::-1]

    train_size = int(len(df) * 0.7)
    train_set = df[0:train_size]
    test_set = df[train_size - seq_length:]

    trainX, trainY = build_dataset(np.array(train_set), seq_length)
    testX, testY = build_dataset(np.array(test_set), seq_length)

    # 텐서로 변환
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    # 텐서 형태로 데이터 정의
    trainDataset = TensorDataset(trainX_tensor, trainY_tensor)
    testDataset = TensorDataset(testX_tensor, testY_tensor)

    return trainX_tensor, trainY_tensor, testX_tensor, testY_tensor, trainDataset, testDataset


# Here we define our model as a class
class LSTMModel(nn.Module):
    def __init__(self, inputXm_dim, inputXl_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.output_dim = output_dim  # 128
        #self.input_dim = input_dim  # <-None
        self.dropout_rate_ph = 0.02

        # base
        #self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim  # <-shape=(None, 12, 128)
        self.num_layers = num_layers  # 128
        self.hidden_dim = hidden_dim  # 128

        #self._hw = 144  # khop, 결과로 나올 시간들. 하루 예측을 하고 싶으니까, 10*144

        # layer
        self.activate_func = tf.nn.relu
        self.lstmXm = nn.LSTM(inputXm_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXl = nn.LSTM(inputXl_dim, hidden_dim, num_layers, batch_first=True)
        #self.fcToLSTM_layer = fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs)  # fc - lstm - fc fore


    def forward(self, xs_input, xm_input, xl_input, hidden_dim, num_layers, output_dim):
        # todo xSinput는 xS의 input 수.. 아마..
        # xSlstmRes,(hnS, cnS) = self.xSlstm(xSinput, hidden_dim, num_layers, output_dim,1)
        # xSlstmRes = self.fcToLSTM_layer(xs_input, hidden_dim, num_layers, output_dim, 0)
        # xMlstmRes = self.fcToLSTM_layer(xm_input, hidden_dim, num_layers, output_dim, 1) # LSTM은 h0, c0있지만 Fc 결과값은 out 만 있으니까!!!
        # xLlstmRes = self.fcToLSTM_layer(xl_input, hidden_dim, num_layers, output_dim, 2)

        xSlstmRes = self.fcToLSTM_layer(xs_input, hidden_dim, num_layers, output_dim, 0)
        xMlstmRes = self.fcToLSTM_layer(xm_input, hidden_dim, num_layers, output_dim, 1) # LSTM은 h0, c0있지만 Fc 결과값은 out 만 있으니까!!!
        xLlstmRes = self.fcToLSTM_layer(xl_input, hidden_dim, num_layers, output_dim, 2)

        #todo 가장 최근 값부터 시작해서, xL만큼 잘라서 concat하고, 그만큼의 날짜 만큼만 사용함.
        #chunk가 아닌 split 사용할 것
        #todo xL만큼의 Y값과 비교해서 LSTM이 역전파 되도록 코딩

        lenXl = xLlstmRes.size()[0]
        xSlstmRes = torch.flip(xSlstmRes,[0])  # tensor reverse
        xSlstmRes = xSlstmRes.split(lenXl, dim = 0)[0]

        xMlstmRes = torch.flip(xMlstmRes, [0])  # tensor reverse
        xMlstmRes = xMlstmRes.split(lenXl, dim=0)[0]

        concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)

        fc = nn.Linear(concatRes.size()[1], output_dim)
        fc1 = fc(concatRes[:, :])

        print('len(xSlstmRes) : ', len(xSlstmRes))
        print('len(xMlstmRes) : ', len(xMlstmRes))
        print('len(xLlstmRes) : ', len(xLlstmRes))
        print('len(concatRes) : ', len(concatRes))
        print('len(fc1) : ', len(fc1))

        return fc1

    def fcToLSTM_layer(self, x, hidden_dim, num_layers, output_dim, whatIs):
        if whatIs == 0:  # short term data
            print('x.size() : ', x.size())
            print("x[0].size() : ", x[0].size())
            print("x[0] : ", x[0])
            print("output_dim : ", output_dim)

            fc = nn.Linear(x.size()[1], output_dim)
            out = fc(x[:, :])
            print("Fc(Xs).shape: ", out.shape)
            print("len(Fc(Xs) : ",len(out))
            #out = self.fc(x[:, -1, :])
            # out.size() -->
        else:  # long term data
            if whatIs == 1 :
                fc = nn.Linear(hidden_dim, output_dim)
                # Initialize hidden state with zeros
                h0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
                # Initialize cell state
                c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()

                # We need to detach as we are doing truncated backpropagation through time (BPTT)
                #out, (hn, cn) = self.lstmXm(x, (h0.detach(), c0.detach()))
                out, _ = self.lstmXm(x)

                out = fc(out[:, :])
                print("lstmXm(Xm).shape: ", out.shape)
                print("len(lstmXm(Xm) : ", len(out))

            else :
                fc = nn.Linear(hidden_dim, output_dim)
                # Initialize hidden state with zeros
                h0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
                # Initialize cell state
                c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()

                # We need to detach as we are doing truncated backpropagation through time (BPTT)

                #out, (hn, cn) = self.lstmXl(x, (h0.detach(), c0.detach()))
                out, _ = self.lstmXl(x)
                out = fc(out[:, :])

                print("lstmXl(Xl).shape: ", out.shape)
                print("len(lstmXl(Xl) : ", len(out))
                #out = fc(out[:, -1, :])

            # Index hidden state of last time step
            #out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
        return out


pd.set_option('display.max_columns', None)
# 데이터 불러오기
df10T = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df10T.set_index('updated', inplace=True)
df1D = pd.read_csv('../data/dailyPred/target102_1D_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df1D.set_index('updated', inplace=True)

## ARIMA
# df1D_arima = df1D
# scaler = MinMaxScaler()
# df1D_arima["power_value"] = scaler.fit_transform(df1D_arima["power_value"].values.reshape(-1, 1))
# df1D_arima = df1D_arima["power_value"]
# #df1D_arima = df1D_arima[::-1]
#
# model  = sm.tsa.arima.ARIMA(df1D_arima,order = (2,1,2),freq = 'D',missing = "drop")
# model_fit = model.fit()
# print("ARIMA model_fit.summary() : ")
# print(model_fit.summary())
#
# preds = model_fit.predict()
# print(preds)



trainXs_tensor, trainYs_tensor, testXs_tensor, testYs_tensor, trainDatasetXs, testDatasetXs = datasetXsml(df1D, 1)
trainXm_tensor, trainYm_tensor, testXm_tensor, testYm_tensor, trainDatasetXm, testDatasetXm = datasetXsml(df1D, 7)
trainXl_tensor, trainYl_tensor, testXl_tensor, testYl_tensor, trainDatasetXl, testDatasetXl = datasetXsml(df1D, 28)


# #ARIMA?
trainXs2_tensor, trainYs2_tensor, testXs2_tensor, testYs2_tensor, trainDatasetXs2, testDatasetXs2 = datasetXsml(df10T, 144)
# datasetXs = trainDatasetXs

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
print("model : ", model)
print("len(list(model.parameters()))", len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
#####################
num_epochs = 200
hist = np.zeros(num_epochs)

# Number of steps to unroll
# look_back = 7
# seq_dim = look_back - 1

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()
    # Forward pass
    #totalLstmFc = model(trainXs2_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
    y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
    print('totalLstmFc_len : ', len(y_train_pred))
    print('totalLstmFc.size : ', y_train_pred.size())
    print('totalLstmFc[0] : ', y_train_pred[0])
    #todo ARIMA랑 concat 후, Xs의 Y 값과 일치하는지 확인. Y_train_pred 도 그러면 dimension 같아야 겠지~

    y_train = torch.flip(len(y_train_pred),[0])
    y_train = torch.flip(len(y_train_pred),[0])
    loss = loss_fn(totalLstmFc, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else t hey will accumulate between epchs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

## Train Fitting1
plt.title('graph1')
plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Real")
plt.legend()
plt.show()

np.shape(y_train_pred)
# make predictions
y_test_pred = model(x_test)
plt.title('graph2')
plt.plot(y_test_pred.detach().numpy(), label="Preds")
plt.plot(y_test.detach().numpy(), label="Real")
plt.legend()
plt.show()

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




