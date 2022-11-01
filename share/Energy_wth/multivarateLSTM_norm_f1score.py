# -*- coding: utf-8 -*-

import sys

import numpy as np
import sklearn.metrics
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 결측치 선형 보간, 900개 데이터에 대해 학습 후 남은 데이터에 대해 하루씩 예측 후 테스트 결과 확인, norm

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

df = pd.read_csv('./data/pv_weather.csv')
print(df.info())

#결측치 보간
for col in df.columns:
    df_intp_linear = df.interpolate()
    df[col] = df_intp_linear[col]
print(df.info())

# in_cols = ['power_value','temp_mean','temp_min','temp_max','humidity_value_avg','humidity_value_min','weather_warning']
in_cols = ['power_value','temp_mean','temp_min','temp_max','weather_warning']
out_cols = ['power_value']  #예측 대상

# choose a number of time steps
n_steps_in, n_steps_out = 30, 1 # 이전 30일 보고 하루 예측

# ==============================================================================
# Preparing Model for 'Low'=======================================================
j = 0

# Scaling dataset
print(df.head())
for col in in_cols:
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

dataset_low = np.empty((df[out_cols[j]].values.shape[0], 0))
for i in range(len(in_cols)):
    dataset_low = np.append(dataset_low, df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0], 1), axis=1)
dataset_low = np.append(dataset_low, df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0], 1), axis=1)
scaled_data = dataset_low

# convert into input/output
train_set_size = 900
x_train, y_train = split_sequences(scaled_data[:train_set_size], n_steps_in, n_steps_out)
# train_set_size = int(0.2 * scaled_data.shape[0])

x_test, y_test = split_sequences(scaled_data[train_set_size:-1, :], n_steps_in, n_steps_out)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(), x_train.size())

# Build model
##################################################

input_dim = 5 # feature 개수 - power value 까지
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

# optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model)
# print(len(list(model.parameters())))
#
# for i in range(len(list(model.parameters()))):
#     print(list(model.parameters())[i].size())

# Train model
##################################################################

hist = np.zeros(num_epochs)
#
# for t in range(num_epochs):
#     # Initialise hidden state
#     # Forward pass
#     y_train_pred = model(x_train)
#     loss = loss_fn(y_train_pred, y_train)
#
#     if t % 10 == 0 and t != 0:
#         print("Epoch ", t, "MSE: ", loss.item())
#     hist[t] = loss.item()
#
#     # Zero out gradient, else they will accumulate between epochs
#     optimiser.zero_grad()
#
#     # Backward pass
#     loss.backward()
#
#     # Update parameters
#     optimiser.step()


model = torch.load("./model/model_e2000_good.pt")

# make predictions
y_test_pred = model(x_test)


def F1scoreLevel( yTrue, yPred,) :

    yTrue = sum(yTrue.tolist(), [])
    yPred = sum(yPred.tolist(),[])

    yDf = pd.DataFrame({"yTrue" : yTrue,
                        "yPred" : yPred}).diff(axis=0, periods=1)
    yDf.iloc[0] = 0

    yTmax = yDf["yTrue"].max()
    yTmin = yDf["yTrue"].min()

    print(type(yTmax))
    print(yTmax)
    print(yTmin)

    y25 = yDf["yTrue"].describe().iloc[4]
    y50 = yDf["yTrue"].describe().iloc[5]
    y75 = yDf["yTrue"].describe().iloc[6]

    yTrueLevel = []
    for i in range(len(yDf)) :
        if yDf["yTrue"].iloc[i] <= y25 :
            yTrueLevel.append(0)
        elif yDf["yTrue"].iloc[i] <= y50 :
            yTrueLevel.append(1)
        elif yDf["yTrue"].iloc[i] <= y75:
            yTrueLevel.append(2)
        else :
            yTrueLevel.append(3)
    yDf["yTrueLevel"] = yTrueLevel


    yPredLevel = []
    for i in range(len(yDf)) :
        if yDf["yPred"].iloc[i] <= y25 :
            yPredLevel.append(0)
        elif yDf["yPred"].iloc[i] <= y50 :
            yPredLevel.append(1)
        elif yDf["yPred"].iloc[i] <= y75:
            yPredLevel.append(2)
        else :
            yPredLevel.append(3)
    yDf["yPredLevel"] = yPredLevel

    return yDf

yDf = F1scoreLevel(y_test, y_test_pred)
yDf.to_csv("./data/yDf_f1.csv")

f1_y_true = yDf["yTrueLevel"].values.tolist()
f1_y_pred = yDf["yPredLevel"].values.tolist()

print(confusion_matrix(f1_y_true, f1_y_pred, labels=[0,1,2,3]))
print(classification_report(f1_y_true, f1_y_pred))


sys.exit()





#
# # torch.save(model, "./model/model_e2000.pt")
#
#
# print("MSE loss---------")
# trainScore = loss_fn(y_train, y_train_pred)
# print('Train MSE Score: %.8f' % (trainScore))
# testScore = loss_fn(y_test, y_test_pred)
# print('Test MSE Score: %.8f' % (testScore))
#
# print("RMSE loss--------")
# trainScore = math.sqrt(loss_fn(y_train, y_train_pred))
# print('Train RMSE Score: %.8f' % (trainScore))
# testScore = math.sqrt(loss_fn(y_test, y_test_pred))
# print('Test RMSE Score: %.8f' % (testScore))
#
# print("R2 Score--------")
# trainScore = r2_score(y_train.detach().numpy(), y_train_pred.detach().numpy())
# print('Train R2 Score: %.8f' % (trainScore))
# testScore = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())
# print('Test R2 Score: %.8f' % (testScore))
#
#
# plt.clf()
# plt.plot(y_train_pred.detach().numpy(), label="Preds")
# plt.plot(y_train.detach().numpy(), label="Data")
# plt.legend()
# plt.savefig("./output/normPv-train.png")
# # plt.show()
#
# plt.clf()
# plt.plot(hist, label="Training loss")
# plt.legend()
# plt.savefig("./output/normPv-training_loss.png")
# # plt.show()
#
#
#
# plt.clf()
# plt.plot()
# plt.plot(y_test_pred.detach().numpy(), label="Preds")
# plt.plot(y_test.detach().numpy(), label="Data")
# plt.legend()
# plt.savefig("./output/normPv-pred.png")
# # plt.show()


