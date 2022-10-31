#ref) https://github.com/kianData/PyTorch-Multivariate-LSTM

# -*- coding: utf-8 -*-

import sys

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

# 결측치 선형 보간, 900개 데이터에 대해 학습 후 남은 데이터에 대해 하루씩 예측 후 테스트 결과 확인

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences) - 1):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix + 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# reading data frame ==================================================
# df = pd.read_csv('./data/goldETF.csv')
df = pd.read_csv('./data/result_pv_1031_2.csv')
print(df.info())

#결측치 보간
for col in df.columns:
    df_intp_linear = df.interpolate()
    df[col] = df_intp_linear[col]
print(df.info())

# in_cols = ['power_value','temp_mean','temp_min','temp_max','humidity_value_avg','humidity_value_min','weather_warning']
in_cols = ['temp_mean','temp_min','temp_max','humidity_value_avg','humidity_value_min','weather_warning'] # target을 왜 feature로 넣어야 하나..?
out_cols = ['power_value']  #예측 대상

# choose a number of time steps
n_steps_in, n_steps_out = 30, 1 # 이전 30일 보고 하루 예측
# n_steps_in, n_steps_out = 30, 7 # 이전 30일 보고 7일 예측

# ==============================================================================
# Preparing Model for 'Low'=======================================================
j = 0

# Scaling dataset
for col in in_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

dataset_low = np.empty((df[out_cols[j]].values.shape[0], 0))
for i in range(len(in_cols)):
    dataset_low = np.append(dataset_low, df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0], 1), axis=1)
dataset_low = np.append(dataset_low, df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0], 1), axis=1)
scaled_data = dataset_low


# convert into input/output
x_train, y_train = split_sequences(scaled_data, n_steps_in, n_steps_out)

train_set_size = 900
x_test, y_test = split_sequences(scaled_data[-train_set_size:-1, :], n_steps_in, n_steps_out)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(), x_train.size())

# Build model
##################################################

input_dim = 6 # feature 개수 - power value 까지
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 2000


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model)
# print(len(list(model.parameters())))



for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Train model
##################################################################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
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

torch.save(model, "./model/multi_torch2_withoutPV.pt")


# make predictions
y_test_pred = model(x_test)

trainScore = loss_fn(y_train, y_train_pred)
print('Train Score: %.8f MSE' % (trainScore))
testScore = loss_fn(y_test, y_test_pred)
print('Test Score: %.8f MSE' % (testScore))




plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.savefig("./output/withoutPv-train.png")
# plt.show()


plt.plot(hist, label="Training loss")
plt.legend()
plt.savefig("./output/withoutPv-training_loss.png")
# plt.show()

plt.plot(y_test_pred.detach().numpy(), label="Preds")
plt.plot(y_test.detach().numpy(), label="Data")
plt.legend()
plt.savefig("./output/withoutPv-pred.png")
# plt.show()








sys.exit()

ytrainpred_copies_array = np.repeat(y_train_pred.detach().numpy(),input_dim+1, axis=-1)
y_train_pred=scaler.inverse_transform(np.reshape(ytrainpred_copies_array,(len(y_train_pred),input_dim+1)))[:,0]

ytrain_copies_array = np.repeat(y_train.detach().numpy(),input_dim+1, axis=-1)
y_train=scaler.inverse_transform(np.reshape(ytrain_copies_array,(len(y_train),input_dim+1)))[:,0]

ytestpred_copies_array = np.repeat(y_test_pred.detach().numpy(),input_dim+1, axis=-1)
y_test_pred=scaler.inverse_transform(np.reshape(ytestpred_copies_array,(len(y_test_pred),input_dim)))[:,0]

ytest_copies_array = np.repeat(y_test.detach().numpy(),input_dim, axis=-1)
y_test=scaler.inverse_transform(np.reshape(ytest_copies_array,(len(y_test),input_dim)))[:,0]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:], y_train_pred[:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:], y_test_pred[:]))
print('Test Score: %.2f RMSE' % (testScore))


print("test loss : ",loss_fn(y_test_pred, y_test))
print("y_train[0] : ",y_train[0])
# invert predictions
# y_train_pred = scaler_all.inverse_transform(y_train_pred.detach().numpy()) #input dim이 3인데 ypred 값은 1개만 나오니까


print(y_test_pred)
print(len(y_test_pred))
print(len(y_test_pred[0]))

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.plot(y_train_pred)
plt.plot(y_train)
plt.show()

# plot baseline and predictions
plt.figure(figsize=(15, 8))
plt.plot(y_test_pred)
plt.plot(y_test)
plt.show()