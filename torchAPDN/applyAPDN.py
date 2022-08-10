# ref) https://eunhye-zz.tistory.com/8#google_vignette

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import sys
import math, time
import itertools
from datetime import datetime
from operator import itemgetter
from math import sqrt
from torch.autograd import Variable
import pickle

device = torch.device('cpu')

'''
    이전 7일 간의 데이터를 기반으로 다음 날의 종가를 예측함 Sequence = 7, output dimension = 1
    예측하기 위해 사용하는 데이터는 시가, 종가 등 총 5개의 column(Input demension)

'''
pd.set_option('display.max_columns', None)
''' 학습/테스트 데이터 분할 '''
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#데이터 불러오기
df = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding ='utf-8', ) #10분 단위의 계측치
df.set_index('updated', inplace=True)
#결측치 있어서 보간 필요(index를 datatime으로 해서 그런지는 모름 이유 파악 X) -> target102_3H_diff.CSV 는 이미 끝난 파일
#df_intp_linear = df.interpolate()
#df['power_value'] = df_intp_linear[['power_value', 'pw_diff']]
#df = df[['power_value', 'pw_diff']]

df = df[['power_value', ]]
#print(len(df)) #8860



scaler = MinMaxScaler()
df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));  #220
    train_set_size = data.shape[0] - (test_set_size); #881

    x_train = data[:train_set_size, :-1, :] #(881,6,1) 이전 7일까지의 값을 사용하니까
    y_train = data[:train_set_size, -1, :]  #(881,1)

    x_test = data[train_set_size:, :-1]  # (220,1)
    y_test = data[train_set_size:, -1, :]  #(881,1)

    return [x_train, y_train, x_test, y_test]


look_back = 7  # choose sequence length
x_train, y_train, x_test, y_test = load_data(df, look_back)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)
# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(),x_train.size())

# Build model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
sequence_num1 = 0 # xS
sequence_num2 = 1 # xM, xL

# Here we define our model as a class




model = APDN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
#####################
num_epochs = 100
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
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

plt.title('graph3')
plt.plot(y_test_pred,label="Preds")
plt.plot(y_test, label = "Real")
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
#plt.plot(y_test.detach().numpy(), label="Data")
plt.legend()
plt.show()