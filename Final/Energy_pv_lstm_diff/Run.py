import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from .config import lstm_parse
from matplotlib import font_manager, rc
from .LSTMModel import LSTM

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import math, time
from math import sqrt

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import r2_score

# 10T data -> 1D data resampling & 선형 보간
def resampleFreq(args, df) :
    df.set_index(args.date_column, inplace=True)
    resultDf = df.resample(args.freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf[args.target_name] = df_intp_linear[args.target_name]
    resultDf = resultDf.diff(axis=0, periods=1)
    resultDf.iloc[0] = 0
    # scaler = MinMaxScaler()
    # resultDf[args.target_name] = scaler.fit_transform(resultDf[args.target_name].values.reshape(-1, 1))

    return resultDf

def load_data(stock, look_back):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])


    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))  # 220
    train_set_size = data.shape[0] - (test_set_size)  # 881

    x_train = data[:train_set_size, :-1, :]  # (757, 27, 1) 이전 7일까지의 값을 사용하니까
    y_train = data[:train_set_size, -1, :]  # (757, 1)

    x_test = data[train_set_size:, :-1]  # (189, 27, 1) #그냥 반복하고 있는데, 실제로는 매번 예측한 값을 반복해서 더하도록 변경해야 함
    y_test = data[train_set_size:, -1, :]  # (189,1)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]



def Train(args,model, x_train, y_train) :
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = np.zeros(args.num_epochs)

    for t in range(args.num_epochs):
        # model.hidden = model.init_hidden()
        # Forward pass
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    # y_train = scaler.inverse_transform(y_train.detach().numpy())
    trainScore = loss_fn(y_train,y_train_pred)
    print('Train MSE Score: %.10f' % (trainScore))


    trainScore = math.sqrt(loss_fn(y_train[:, 0], y_train_pred[:, 0]))
    print('Train RMSE Score: %.10f ' % (trainScore))

    resultdf = pd.DataFrame({"TrainY": [*(y_train[:, -1].tolist())],
                             "TrainPred": [*(y_train_pred[:, 0].tolist())]})
    print("train-train Pred")
    print(resultdf)

    R2Score = sklearn.metrics.r2_score(y_train, y_train_pred)
    print('Train R2 Score : %.10f' %(R2Score))


    torch.save(model,  args.model_save)
    return y_train_pred, y_train,model

def Predict( args,model, x_test, y_test ) :
    y_test_pred = model(x_test)
    loss_fn = torch.nn.MSELoss()
    # invert predictions

    trainScore = loss_fn(y_test,y_test_pred)
    print('Test MSE Score: %.10f MSE' % (trainScore))

    # y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    # y_test = scaler.inverse_transform(y_test.detach().numpy())

    testScore = math.sqrt(loss_fn(y_test,y_test_pred))
    print('Test RMSE Score: %.2f RMSE' % (testScore))

    resultdf = pd.DataFrame({"TestY": [*(y_test[:, -1].tolist())],
                             "TestPred": [*(y_test_pred[:, 0].tolist())]})
    print("test-test Pred")
    print(resultdf)

    R2Score = sklearn.metrics.r2_score(y_test, y_test_pred)
    print('Test R2 Score : %.10f' %(R2Score))

    # with open(args.pred_result, 'w') as csv_file:
    #     yDf[-30 : ].to_csv(path_or_buf=csv_file)

    return y_test_pred, y_test



def Visualize(y_pred, real, path, title) :
    plt.gca()
    plt.gcf()
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 10))
    plt.title(title)

    plt.plot(y_pred, label="Preds")
    plt.plot(real, label="Real")
    plt.legend()
    # plt.show()
    plt.savefig(path)


def main():
    parser = argparse.ArgumentParser(description='Embedding arguments')
    lstm_parse(parser)
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    df1D = pd.read_csv(args.path, parse_dates=[args.date_column], encoding='utf-8', )
    resultDf = resampleFreq(args, df1D)  # 일 단위 데이터로 변환 및 결측치 선형 보간


    print(resultDf.describe())

    sys.exit()







    x_train, y_train, x_test, y_test = load_data(resultDf, args.look_back)

    if args.state == "train" :

        model = LSTM(input_dim=args.input_dim,
                     hidden_dim=args.hidden_dim,
                     output_dim=args.output_dim,
                     num_layers=args.num_layers)

        y_train_pred, y_train, model = Train(args, model, x_train, y_train)

        Visualize(y_train_pred.detach().numpy(), y_train.detach().numpy(), args.result_train, "train")

        resultdf = pd.DataFrame({"TrainY": [*(y_train[:, -1].tolist())],
                                 "TrainPred": [*(y_train_pred[:, 0].tolist())]})
        print("train-train Pred")
        print(resultdf)


    else :
        model = torch.load(args.model_save)
        y_test_pred, y_test = Predict(args, model, x_test, y_test)
        Visualize(y_test_pred.detach().numpy(), y_test.detach().numpy(), args.result_pred, "test")


        resultdf = pd.DataFrame({"TestY": [*(y_test.tolist())],
                                 "TestPred": [*(y_test_pred.tolist())]})
        print("test-test Pred")
        print(resultdf)



if __name__ == '__main__':
    main()
