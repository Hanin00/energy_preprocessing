#ref) https://direction-f.tistory.com/23
# https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import datetime


'''
    개선에 관한 의견
    1. 누적 값이고, 각 3분 정도의 간격이 있다고 해도 항상 같은게 아니니까
        누적 값 간 증가량을 특징으로 하면 더 잘 나올 듯(빈 값들은 보간하고)
'''

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#데이터 불러오기
df = pd.read_csv('../data/dailyPred/target102_3H_diff.csv', parse_dates=['updated'], encoding ='utf-8', )
#df = pd.read_csv('./data/target102_3H_diff.csv',parse_dates=['updated'],  encoding = 'utf-8', )
df.set_index('updated', inplace=True)

#결측치 있어서 보간 필요(index를 datatime으로 해서 그런지는 모름 이유 파악 X)
# df_intp_linear = df.interpolate()
# data_ski = df_intp_linear[["power_value"]]
# data_ski = df[['power_value','pw_diff']]
data_ski = df[['power_value']]

## scaling
scaler = MinMaxScaler()
data_ski["power_value"] = scaler.fit_transform(data_ski["power_value"].values.reshape(-1, 1))


# 일 별 예측량은 0일때 시작해서 한 칸씩 미루면 되는 건가..?
#window_size = 학습시 고려하는 이전 일자
## sequence data
def make_dataset(data, window_size=7):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(data.iloc[i + window_size]))

    return np.array(feature_list), np.array(label_list)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

data_X, data_Y = make_dataset(data_ski)  #(1108,  1)

train_data, train_label = data_X[:-300, ], data_Y[:-300, ]  #(788,20,1),(788, 1)
test_data, test_label = data_X[-300:, ], data_Y[-300:,]  #(300, 20, 1), (300, 1)


## tensor set
X_train = torch.from_numpy(train_data).float()
y_train = torch.from_numpy(train_label).float()

X_test = torch.from_numpy(test_data).float()
y_test = torch.from_numpy(test_label).float()

# Model
class LSTM_(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        #out, (hn, cn) = self.lstm(x, (h0, c0))
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


    # def forward(self, x):
    #     h0, c0 = self.init_hidden(x) #
    #     out, (hn, cn) = self.lstm(x, (h0, c0))
    #     out = self.fc(out[:, -1, :])
    #     return out

    # def init_hidden(self, x):
    #     # self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
    #     # self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
    #     self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
    #     self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
    #     return h0, c0

#Train loop
model = LSTM_(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)
loss_fn = torch.nn.MSELoss(reduction="sum")
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(200)

for t in range(200):
    # Forward pass
    y_train_pred = model(X_train)

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
plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Real")
plt.legend()
plt.show()

## Test Fitting
y_test_pred = model(X_test)
plt.plot(y_test_pred.detach().numpy(), label="Preds")
plt.plot(y_test.detach().numpy(), label="Real")
plt.legend()
plt.show()

## Test Fitting-inverser scaling
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())
plt.plot(y_test_pred,label="Preds")
plt.plot(y_test, label = "Real")
plt.legend()
plt.show()

test_seq = X_test[:1]  ## X_test에 있는 데이터중 첫번째것을 가지고 옮
preds = []

for _ in range(len(X_test)):
    # model.init_hidden(test_seq)
    y_test_pred = model(test_seq)

    pred = y_test_pred.item()
    preds.append(pred)
    new_seq = test_seq.numpy()
    new_seq = np.append(new_seq, pred)
    new_seq = new_seq[1:]  ## index가 0인 것은 제거하여 예측값을 포함하여 20일치 데이터 구성
    test_seq = torch.from_numpy(new_seq).view(1, 7, 1).float()







