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
from torch.autograd import Variable


def slicerFreq(df, freq, dateColumn) :
    df.set_index(dateColumn, inplace=True)
    resultDf = df.resample(freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf["power_value"] = df_intp_linear[["power_value"]]
    return resultDf


pd.set_option('display.max_columns', None)
# 데이터 불러오기
# df = pd.read_csv('./data/target102_3H_diff.csv',parse_dates=['updated'],  encoding = 'utf-8', )
df = pd.read_csv('./data/new_total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )  # <- 10분 단위 데이터
#일단위 데이터로 변경

dateColumn = 'updated'
freq = 'D'

resultDf = slicerFreq(df, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간

look_back = 7  # choose sequence length

scaler = MinMaxScaler()
resultDf["power_value"] = scaler.fit_transform(resultDf["power_value"].values.reshape(-1, 1))


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


look_backXs = 1  # choose sequence length
look_backXm = 7  # choose sequence length <- 하루. 1008이 일주일인데, 이렇게 하니까 OoM 오류 남
look_backXl = 28  # choose sequence length <- 하루. 1008이 일주일인데, 이렇게 하니까 OoM 오류 남
xs_train, ys_train, xs_test, ys_test = load_data(resultDf, look_backXs)
xm_train, ym_train, xm_test, ym_test = load_data(resultDf, look_backXm)
xl_train, yl_train, xl_test, yl_test = load_data(resultDf, look_backXl)
# print('x_train.shape = ', x_train.shape)
# print('y_train.shape = ', y_train.shape)
# print('x_test.shape = ', x_test.shape)
# print('y_test.shape = ', y_test.shape)
# make training and test sets in torch
xs_train = torch.from_numpy(xs_train).type(torch.Tensor)
xs_test = torch.from_numpy(xs_test).type(torch.Tensor)
ys_train = torch.from_numpy(ys_train).type(torch.Tensor)
ys_test = torch.from_numpy(ys_test).type(torch.Tensor)

xm_train = torch.from_numpy(xm_train).type(torch.Tensor)
xm_test = torch.from_numpy(xm_test).type(torch.Tensor)
ym_train = torch.from_numpy(ym_train).type(torch.Tensor)
ym_test = torch.from_numpy(ym_test).type(torch.Tensor)

xl_train = torch.from_numpy(xl_train).type(torch.Tensor)
xl_test = torch.from_numpy(xl_test).type(torch.Tensor)
yl_train = torch.from_numpy(yl_train).type(torch.Tensor)
yl_test = torch.from_numpy(yl_test).type(torch.Tensor)

# print(y_train.size(), x_train.size())

# Build model
#####################
input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, input_dimXs,input_dimXm, input_dimXl, hidden_dim, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)




        self.lstmXs = nn.LSTM(input_dimXs, hidden_dim, num_layers, batch_first=True)
        self.lstmXm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstmXl = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fcXs = nn.Linear(hidden_dim, output_dim)
        self.fcXm = nn.Linear(hidden_dim, output_dim)
        self.fcXl = nn.Linear(hidden_dim, output_dim)

    def forward(self, xs, xm, xl):

        # #xs
        # hs0 = torch.zeros(self.num_layers, xs.size(0), self.hidden_dim).requires_grad_()
        # cs0 = torch.zeros(self.num_layers, xs.size(0), self.hidden_dim).requires_grad_()
        # # outXm, _ = self.lstmXm(xm)
        # outXs, (hn, cn) = self.lstmXs(xs, (hs0.detach(), cs0.detach()))
        #
        # # Index hidden state of last time step
        # # out.size() --> 100, 32, 100
        # # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # outXs = self.fcXs(outXs[:, -1, :])

        #xm
        # Initialize hidden state with zeros
        hm0 = torch.zeros(self.num_layers, xm.size(0), self.hidden_dim).requires_grad_()
        cm0 = torch.zeros(self.num_layers, xm.size(0), self.hidden_dim).requires_grad_()
        # outXm, _ = self.lstmXm(xm)
        outXm, (hn, cn) = self.lstmXm(xm, (hm0.detach(), cm0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        outXm = self.fcXm(outXm[:, -1, :])

        #xl
        hl0 = torch.zeros(self.num_layers, xl.size(0), self.hidden_dim).requires_grad_()
        cl0 = torch.zeros(self.num_layers, xl.size(0), self.hidden_dim).requires_grad_()
        # outXm, _ = self.lstmXm(xm)
        outXl, (hn, cn) = self.lstmXl(xl, (hl0.detach(), cl0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        outXl = self.fcXl(outXl[:, -1, :])

        #Concat
        lenXl = outXl.size()[0]
        # xSlstmRes = torch.flip(outXs, [0])  # tensor reverse
        # xSlstmRes = xSlstmRes.split(lenXl, dim=0)[0]
        xMlstmRes = torch.flip(outXm, [0])  # tensor reverse
        xMlstmRes = xMlstmRes.split(lenXl, dim=0)[0]

        # concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)
        concatRes = torch.cat(( xMlstmRes, outXl), input_dim)

        self.fc = nn.Linear(concatRes.size()[1], output_dim)
        fc1 = self.fc(concatRes[:, :])

        # out.size() --> 100, 10
        return fc1



model = LSTM(input_dim,xs_train.size()[1],xm_train.size()[1],xl_train.size()[1], hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model)
# print(len(list(model.parameters())))
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

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
    yl_train_pred = model(xs_train, xm_train, xl_train)

    loss = loss_fn(yl_train_pred, yl_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

## Train Fitting
plt.title('graph1')
plt.plot(ym_train_pred.detach().numpy(), label="Preds")
plt.plot(ym_train.detach().numpy(), label="Real")
plt.legend()
plt.show()

np.shape(ym_train_pred)
# make predictions
ym_test_pred = model(xs_test, xm_test, xl_test)
plt.title('graph2')
plt.plot(ym_test_pred.detach().numpy(), label="Preds")
plt.plot(ym_test.detach().numpy(), label="Real")
plt.legend()
plt.show()


sys.exit()
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



# ARIMA 보다 더 나은건가 함 봐볼게요..