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

pd.set_option('display.max_columns', None)
# 데이터 불러오기
df = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df.set_index('updated', inplace=True)

scaler = MinMaxScaler()
df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))

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
def load_data(meter, look_backTs, look_backTm, look_backTl):
    data_raw = meter.values  # convert to numpy array
    x = []
    x_skip_tm = []
    x_skip_tl = []
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_backTs):
        x.append(data_raw[index: index + look_backTs])
    for index in range(len(data_raw) - look_backTm):
        x_skip_tm.append(data_raw[index: index + look_backTm])
    for index in range(len(data_raw) - look_backTl):
        x_skip_tl.append(data_raw[index: index + look_backTl])

    test_set_size = int(np.round(0.2 * x.shape[0]))  # 220
    train_set_size = x.shape[0] - (test_set_size)  # 881

    # x_train = [x, x_skip_Tm, x_skip_Tl, label_s]
    # x_train = [x[:train_set_size, :-1, :], x_skip_ts[:train_set_size, :-1, :],  ]#(881,6,1) 이전 7일까지의 값을 사용하니까



    x_train = x[:train_set_size, :-1, :]  #
    y_train = x[:train_set_size, -1, :]  #

    x_test = data[train_set_size:, :-1]  #
    y_test = data[train_set_size:, -1, :]  #

    return [x_train, y_train, x_test, y_test]




look_backTs = 144  # choose sequence length <- 10min * 144 = 1440 = 1Day
look_backTm = 144*7  # choose sequence length <- 10min * 144 * 7 = 7Day <- 1008
look_backTl = 144*7*4  # choose sequence length <- 10min * 144 * 7* 4= 28Day  <- 4032

x_train, y_train, x_test, y_test = load_data(df["power_value"], look_backTs, look_backTm,
                                             look_backTl)  # 6day is feature, cause use for 1 week is Long term
# print('x_train.shape = ', x_train.shape)  # (127570, 6, 4)
# print('y_train.shape = ', y_train.shape) #(127570, 4)
# print('x_test.shape = ', x_test.shape)  #(31892, 6, 4)
# print('y_test.shape = ', y_test.shape)  # (31892, 4)
# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(), x_train.size())

# Build model
#####################
input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1

# Here we define our model as a class
class PredictModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, X):
        super(PredictModel, self).__init__()
        self.output_dim = output_dim  # 128
        self.input_dim = input_dim  # <-None
        self.dropout_rate_ph = 0.02

        # data
        self.X_short = 144
        self.X_mid = 7
        self.X_long = 28

        # base
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim  # <-shape=(None, 12, 128)
        self.num_layers = num_layers  # 128
        self.hidden_dim = hidden_dim  # 128

        # layer
        self.activate_func = tf.nn.relu
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fcToLSTM_layer = fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs)  # fc - lstm - fc fore
        # self. # ar + fc

        # self.X_mid_skip_ph = tf.compat.v1.placeholder(self.float_dtype,shape=[None, self.n, self.T, self.D])
        # shape -> [batch_size, T, v] #short term
        x_short_input = self.input_layer(self.X)
        x_mid_input = self.input_layer(self.X_mid)  # mid term,
        x_long_input = self.input_layer(self.X_long)  # long term,

        self.xSlstm, (hnS, cnS) = self.fcToLSTM_layer(x_short_input, hidden_dim, num_layers, 1)
        self.xMlstm, (hnM, cnM) = self.fcToLSTM_layer(x_mid_input, hidden_dim, num_layers, 0)
        self.xLlstm, (hnL, cnL) = self.fcToLSTM_layer(x_long_input, hidden_dim, num_layers, 0)



    def forward(self, xSinput, xMinput, xLinput, hidden_dim, num_layers, output_dim):
        # todo xSinput는 xS의 input 수.. 아마..
        # xSlstmRes,(hnS, cnS) = self.xSlstm(xSinput, hidden_dim, num_layers, output_dim,1)
        xSlstmRes, (hnS, cnS) = fcToLSTM_layer(xSinput, hidden_dim, num_layers, output_dim, 1)
        xMlstmRes, (hnM, cnM) = fcToLSTM_layer(xMinput, hidden_dim, num_layers, output_dim, 0)
        xLlstmRes, (hnL, cnL) = fcToLSTM_layer(xLinput, hidden_dim, num_layers, output_dim, 0)
        concatRes = torch.cat((xSlstmRes, xMlstmRes, xLlstmRes), self.input_dim)
        fc1 = self.fc(concatRes[:, -1, :])

        # horisontal - predict?
        for xs in xSinput:
            ar_input = self.X_ph[-self._hw:]

            h0 = torch.zeros(num_layers, xs.size(0), hidden_dim).requires_grad_()
            # Initialize cell state
            c0 = torch.zeros(num_layers, x.size(0), hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # generate predictions

        return fc1

    def fcToLSTM_layer(self, x, hidden_dim, num_layers, whatIs, scope='fcToLSTM'):
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


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

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
    y_train_pred = model(x_train)

    loss = loss_fn(y_train_pred, y_train)
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
