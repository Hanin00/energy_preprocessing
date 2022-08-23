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
def datasetXsml(df, seq_length, ):
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


class LSTMnet(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTMnet, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)

        x = self.fc(x[:, -1])
        return x


def LSTMtrain_model(model, train_df, num_epochs=None, lr=None, verbose=10, patience=10):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    nb_epochs = num_epochs

    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples

            # seq별 hidden state reset
            model.reset_hidden_state()

            # H(x) 계산
            outputs = model(x_train)

            # cost 계산
            loss = criterion(outputs, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch

        train_hist[epoch] = avg_cost

        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):

            # loss가 커졌다면 early stop
            if train_hist[epoch - patience] < train_hist[epoch]:
                print('\n Early Stopping')
                break

    return model.eval(), train_hist

pd.set_option('display.max_columns', None)
# 데이터 불러오기
df10T = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df10T.set_index('updated', inplace=True)
df1D = pd.read_csv('../data/dailyPred/target102_1D_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df1D.set_index('updated', inplace=True)


df1D_arima = df1D
scaler = MinMaxScaler()
df1D_arima["power_value"] = scaler.fit_transform(df1D_arima["power_value"].values.reshape(-1, 1))
df1D_arima = df1D_arima["power_value"]
#df1D_arima = df1D_arima[::-1]

model  = sm.tsa.arima.ARIMA(df1D_arima,order = (2,1,2),freq = 'D',missing = "drop")
model_fit = model.fit()
print("ARIMA model_fit.summary() : ")
print(model_fit.summary())


preds = model_fit.predict()
print(preds)


#todo LSTM 모듈을 하나로 만들고 ARIMA 출력값과 Concat 하고, Predict 결과 LSTM 학습에 반영
#todo LSTM 모듈의 입력값이 Xm, Xl이 맞는지 확인하고 Xs가 FC가 되는지 확인
#todo 각 결과값의 형태 확인(shape이든 size든)


trainXs_tensor, trainYs_tensor, testXs_tensor, testYs_tensor, trainDatasetXs, testDatasetXs = datasetXsml(df10T, 144)
trainXm_tensor, trainYm_tensor, testXm_tensor, testYm_tensor, trainDatasetXm, testDatasetXm = datasetXsml(df1D, 7)
trainXl_tensor, trainYl_tensor, testXl_tensor, testYl_tensor, trainDatasetXl, testDatasetXl = datasetXsml(df1D, 28)

datasetXs = trainDatasetXs

batch = 144
seq_length = 144
# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloaderXs = DataLoader(datasetXs, batch_size=batch, shuffle=True, drop_last=True)

data_dim = 144  # 144x10 = 1day
hidden_dim = 144
output_dim = 1
learning_rate = 0.01
nb_epochs = 10
device = 'cpu'


# Xs 모델 학습
lstmNetXs = LSTMnet(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
modelLstmXs, train_histLstmXs = LSTMtrain_model(lstmNetXs, dataloaderXs, num_epochs=nb_epochs, lr=learning_rate,
                                                verbose=20, patience=10)


# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.title("Xs")
plt.plot(train_histLstmXs, label="Training loss")
plt.legend()
plt.show()

# Xm 모델 학습

data_dim = 7  # 1DAY * 7 = 1 Week
hidden_dim = 7
output_dim = 1
learning_rate = 0.01
nb_epochs = 10
device = 'cpu'

batchXm = 7
seq_lengthXm = 7

datasetXm = trainDatasetXm
# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloaderXm = DataLoader(datasetXm, batch_size=batchXm, shuffle=True, drop_last=True)

lstmNetXm = LSTMnet(data_dim, hidden_dim, seq_lengthXm, output_dim, 1).to(device)
modelLstmXm, train_histLstmXm = LSTMtrain_model(lstmNetXm, dataloaderXm, num_epochs=nb_epochs, lr=learning_rate,
                                                verbose=20, patience=10)

# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.title("Xm")
plt.plot(train_histLstmXm, label="Training loss")
plt.legend()
plt.show()

# Xl
data_dim = 28  # 144x10 = 1day
hidden_dim = 28
output_dim = 1
learning_rate = 0.01
nb_epochs = 10
device = 'cpu'

batchXl = 28
seq_lengthXl = 28
datasetXl = trainDatasetXl
# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloaderXl = DataLoader(datasetXl, batch_size=batchXl, shuffle=True, drop_last=True)

# 모델 학습
lstmNetXl = LSTMnet(data_dim, hidden_dim, seq_lengthXl, output_dim, 1).to(device)
modelLstmXl, train_histLstmXl = LSTMtrain_model(lstmNetXl, dataloaderXl, num_epochs=nb_epochs, lr=learning_rate,
                                                verbose=20, patience=10)

# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.title("Xl")
plt.plot(train_histLstmXl, label="Training loss")
plt.legend()
plt.show()

# 예측 테스트
with torch.no_grad():
    pred = []
    for pr in range(len(testXs_tensor)):
        modelLstmXs.reset_hidden_state()
        predicted = model(torch.unsqueeze(testXs_tensor[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)

    # INVERSE
    pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
    testY_inverse = scaler_y.inverse_transform(testYs_tensor)


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


print('MAE SCORE : ', MAE(pred_inverse, testY_inverse))

'''
    일단 train, test, val을 나누고
    meter : 10분 단위 계측값
    Ts : 144, 하루. short term
    Tm : 7, 일주일, mid term
    Tl : 28, 4주, long term 

    AR 이 D. Generating the Prediction

    len(x) :  159325
    len(x_skip_tm) :  158461
    len(x_skip_tl) :  155437
    len(x[0]) :  144
    len(x_skip_tm[0]) :  1008
    len(x_skip_tl[0]) :  4032
'''


# function to create train, test data given meter data and sequence length
def load_data(meter, look_back):
    data_raw = meter.values  # convert to numpy array
    x = []
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        x.append(data_raw[index: index + look_back])

    test_set_size = int(np.round(0.2 * len(x)))
    print("len(x[0]) : ", len(x))  # 159325
    print("len(x[0]) : ", len(x[0]))  # 144

    print("test_set_size : ", test_set_size)  # 29
    train_set_size = len(x) - test_set_size  # 881
    print("train_set_size : ", train_set_size)  # 29
    # x_train = [x, x_skip_Tm, x_skip_Tl, label_s]
    # x_train = [x[:train_set_size, :-1, :], x_skip_ts[:train_set_size, :-1, :],  ]#(881,6,1) 이전 7일까지의 값을 사용하니까

    print(type(x))
    print(type(train_set_size))
    print()
    x_train = x[:train_set_size, :-1, :]  #
    y_train = x[:train_set_size, -1, :]  #

    x_test = data[train_set_size:, :-1]  #
    y_test = data[train_set_size:, -1, :]  #

    return [x_train, y_train, x_test, y_test]


# totalData = df["power_value"].tolist() #10분 단위 데이터. 전체 데이터 셋
# print(totalData[:10])


look_backTs = 144  # choose sequence length <- 10min * 144 = 1440 = 1Day
look_backTm = 144 * 7  # choose sequence length <- 10min * 144 * 7 = 7Day <- 1008
look_backTl = 144 * 7 * 4  # choose sequence length <- 10min * 144 * 7* 4= 28Day  <- 4032

# Build model

input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1

xSinput, xMinput, xLinput = 114, 144 * 7, 144 * 28
steps = [xSinput, xMinput, xLinput]


# Here we define our model as a class
class PredictModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, steps):
        super(PredictModel, self).__init__()
        self.output_dim = output_dim  # 128
        self.input_dim = input_dim  # <-None
        self.dropout_rate_ph = 0.02

        # data
        self.X_short = steps[0]
        self.X_mid = steps[1]
        self.X_long = steps[2]

        # base
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim  # <-shape=(None, 12, 128)
        self.num_layers = num_layers  # 128
        self.hidden_dim = hidden_dim  # 128

        self._hw = 144  # khop, 결과로 나올 시간들. 하루 예측을 하고 싶으니까, 10*144

        # layer
        self.activate_func = tf.nn.relu
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fcToLSTM_layer = fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs)  # fc - lstm - fc fore
        # self. # ar + fc

        # self.X_mid_skip_ph = tf.compat.v1.placeholder(self.float_dtype,shape=[None, self.n, self.T, self.D])
        # shape -> [batch_size, T, v] #short term
        # x_short_input = self.input_layer(self.X_short)
        # x_mid_input = self.input_layer(self.X_mid)  # mid term,
        # x_long_input = self.input_layer(self.X_long)  # long term,

        # self.xSlstm, (hnS, cnS) = self.fcToLSTM_layer(X_short.size[0], hidden_dim, num_layers, 1)
        # self.xMlstm, (hnM, cnM) = self.fcToLSTM_layer(x_mid_input.size[0], hidden_dim, num_layers, 0)
        # self.xLlstm, (hnL, cnL) = self.fcToLSTM_layer(x_long_input.size[0], hidden_dim, num_layers, 0)

    def forward(self, data, steps, hidden_dim, num_layers, output_dim):
        # todo xSinput는 xS의 input 수.. 아마..
        # xSlstmRes,(hnS, cnS) = self.xSlstm(xSinput, hidden_dim, num_layers, output_dim,1)
        xSlstmRes, (hnS, cnS) = fcToLSTM_layer(steps[0], hidden_dim, num_layers, output_dim, 1)
        xMlstmRes, (hnM, cnM) = fcToLSTM_layer(steps[1], hidden_dim, num_layers, output_dim, 0)
        xLlstmRes, (hnL, cnL) = fcToLSTM_layer(steps[2], hidden_dim, num_layers, output_dim, 0)
        concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)
        fc1 = self.fc(concatRes[:, -1, :])

        print('len(xSlstmRes) : ', len(xSlstmRes))
        print('len(xMlstmRes) : ', len(xMlstmRes))
        print('len(xLlstmRes) : ', len(xLlstmRes))
        print('len(concatRes) : ', len(concatRes))
        print('len(fc1) : ', len(fc1))

        # predict - AR
        # horisontal - predict?
        for xs in xSinput:
            h0 = torch.zeros(num_layers, xs.size(0), hidden_dim).requires_grad_()
            # Initialize cell state
            c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # generate predictions

        return fc1

    def fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs):
        if whatIs == 1:  # short term data
            out = self.fc(x[:, -1, :])
            # out.size() -->
        else:  # long term data
            # Initialize hidden state with zeros
            h0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
            # Initialize cell state
            c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
        return out

#build model
model = PredictModel(input_dim, hidden_dim, num_layers, output_dim, steps)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
#####################
num_epochs = 200
hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim = look_back - 1

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    # model.hidden = model.init_hidden()
    # Forward pass
    y_train_pred = model(data, input_dim, hidden_dim, num_layers, output_dim, steps)

    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else t hey will accumulate between epchs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

## Train Fitting
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

