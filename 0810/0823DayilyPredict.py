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



#1-1. LSTM 일단위 + FC, 데이터 형태 확인
    #1-1-1. FC 할 때 weight 를 한 건지..? 걍 결과값 O
#1-2. LSTM 주단위 + FC, 데이터 형태 확인
#1-3. LSTM 월단위 + FC, 데이터 형태 확인
#2-1. ARIMA 일단위 예측 + FC, 데이터 형태 확인
#3-1. 기존 방법에서 concat 방법 확인
#3-2. 기존 방법에서 Predict 한 방법 확인
#todo 4-1. 모듈화

#todo predict 값에 normalize 해서 plot으로 비교
#todo ARIMA predict reversr 후 lstm 결과값과 concat 하면 됨

# todo 지금 하루 데이터만 뽑는게 맞는지에 대해서도 봐야함


#dataP = 0.7
dataP = 1

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

    train_size = int(len(df) * dataP)
    train_set = df[0:train_size]
    trainDf = pd.DataFrame(train_set, columns=["power_value"])
    trainDf["power_value"] = scaler.fit_transform(trainDf["power_value"].values.reshape(-1, 1))
    trainDf = trainDf["power_value"]
    trainDf = trainDf[::-1]

    test_set = df[train_size - seq_length:]
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
        #self.fcToLSTM_layer = fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs)  # fc - lstm - fc fore


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

pd.set_option('display.max_columns', None)
# 데이터 불러오기
# df10T = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding='utf-8', )
# df10T.set_index('updated', inplace=True)
# df1D = pd.read_csv('../data/dailyPred/target102_1D_diff.csv', parse_dates=['updated'], encoding='utf-8', )
# df1D.set_index('updated', inplace=True)
df1D = pd.read_csv('../data/total_pv.csv', parse_dates=['updated'], encoding='utf-8', )
df1D.set_index('updated', inplace=True)

df1D_arima = df1D
scaler = MinMaxScaler()
df1D_arima["power_value"] = scaler.fit_transform(df1D_arima["power_value"].values.reshape(-1, 1))
df1D_arima = df1D_arima["power_value"]
df1D_arima = df1D_arima[::-1]

#train_size = int(len(df1D_arima) * 0.7)model
train_size = int(len(df1D_arima) * dataP)
train_set_AR = df1D_arima[0:train_size]
train_set_AR = train_set_AR[::-1]  # arima에 쓸 쑤 있는 데이터로 만들기 위해 reverse
#test_set_AR = df1D_arima[train_size - seq_length:]

model  = sm.tsa.arima.ARIMA(train_set_AR, order = (2,1,2),freq = 'D',missing = "drop")
model_fit = model.fit()


preds_arima = model_fit.predict()
preds_arima = torch.FloatTensor(preds_arima.tolist())
preds_arima = torch.flip(preds_arima, [0])
preds_arima = torch.unsqueeze(preds_arima, 1)


trainXs_tensor, trainYs_tensor, testXs_tensor, testYs_tensor, trainDatasetXs, testDatasetXs = datasetXsml(df1D, 1)
trainXm_tensor, trainYm_tensor, testXm_tensor, testYm_tensor, trainDataswetXm, testDatasetXm = datasetXsml(df1D, 7)
trainXl_tensor, trainYl_tensor, testXl_tensor, testYl_tensor, trainDatasetXl, testDatasetXl = datasetXsml(df1D, 28)

# #ARIMA
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

# Train model
#####################
num_epochs = 200
hist = np.zeros(num_epochs)

# Number of steps to unroll
# look_back = 7
# seq_dim = look_back - 1

for t in range(num_epochs):
    y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
    #todo ARIMA랑 concat 후, Xs의 Y 값과 일치하는지 확인. Y_train_pred 도 그러면 dimension 같아야겠지 ~
    y_train = torch.flip(trainYs_tensor, [0])  # tensor reverse
    y_train = y_train.split(len(y_train_pred), dim=0)[0]
    preds_arima = preds_arima.split(len(y_train_pred), dim=0)[0]
    y_train_pred += preds_arima
    y_train_pred = torch.flip(y_train_pred, [0])
    #normalize 가 안돼서 이따위로 나오느 ㄴ걸까..?
    # scaler = MinMaxScaler()
    # df1D_arima["power_value"] = scaler.fit_transform(df1D_arima["power_value"].values.reshape(-1, 1))
    # df1D_arima = df1D_arima["power_value"]
    # df1D_arima = df1D_arima[::-1]

    loss = loss_fn(y_train_pred, y_train)

    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    # Zero out  "gradient, else t hey will accumulate between epchs
    optimiser.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimiser.step()

## Train Fitting1
plt.title('Train')
plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Real")
plt.legend()
plt.show()



np.shape(y_train_pred)
# make predictions
y_train_pred = model(trainXs_tensor, trainXm_tensor, trainXl_tensor, hidden_dim, num_layers, output_dim)
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




