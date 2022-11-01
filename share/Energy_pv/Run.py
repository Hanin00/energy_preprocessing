import sys
import argparse
import pandas as pd
import math, time
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import r2_score

from .LSTMModel import LSTM
from .config import lstm_parse

import warnings
warnings.filterwarnings(action="ignore")


#norm O, 누적값 사용

# 10T data -> 1D data resampling & 선형 보간
def resampleFreq(args, df) :
    df.set_index(args.date_column, inplace=True)
    resultDf = df.resample(args.freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf[args.target_name] = df_intp_linear[args.target_name]
    scaler = MinMaxScaler()
    resultDf[args.target_name] = scaler.fit_transform(resultDf[args.target_name].values.reshape(-1, 1))
    return resultDf, scaler

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

    return x_train, y_train, x_test, y_test

def Train(args,model, x_train, y_train, scaler) :
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    hist = np.zeros(args.num_epochs)

    for t in range(args.num_epochs):
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model,  args.model_save)

    mseScore = loss_fn(y_train, y_train_pred)
    rmseScore = math.sqrt(loss_fn(y_train, y_train_pred))
    R2Score = r2_score(y_train.detach().numpy(), y_train_pred.detach().numpy())

    print('Train MSE Score: %.8f' % (mseScore))
    print('Train RMSE Score: %.8f' % (rmseScore))
    print('Train R2 Score: %.8f' % (R2Score))

    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())

    return model, y_train, y_train_pred

def Predict( args,model, x_test, y_test, scaler ) :
    loss_fn = torch.nn.MSELoss(size_average=True)

    y_test_pred = model(x_test)
    mseScore = loss_fn(y_test, y_test_pred)
    rmseScore = math.sqrt(loss_fn(y_test, y_test_pred))
    R2Score = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())

    print('Test MSE Score: %.8f' % (mseScore))
    print('Test RMSE Score: %.8f' % (rmseScore))
    print('Test R2 Score: %.8f' % (R2Score))

    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    yDf = pd.DataFrame({"y_pred": y_test_pred[:, 0],
                        "y_test": y_test[:, 0]})

    with open(args.pred_result, 'w') as csv_file:
        yDf.to_csv(path_or_buf=csv_file)

    return y_test_pred, y_test, mseScore, rmseScore, R2Score

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
    resultDf, scaler = resampleFreq(args, df1D)  # 일 단위 데이터로 변환 및 결측치 선형 보간

    x_train, y_train, x_test, y_test = load_data(resultDf, args.look_back)  # step

    if args.state == "train" :

        model = LSTM(input_dim=args.input_dim,
                     hidden_dim=args.hidden_dim,
                     output_dim=args.output_dim,
                     num_layers=args.num_layers)

        model, y_train, y_train_pred  = Train(args, model, x_train, y_train, scaler)
        Visualize(y_train_pred, y_train, args.result_train, "train")

    elif args.state == "test" :
        x_train, y_train, x_test, y_test = load_data(resultDf, args.look_back)
        model = torch.load(args.model_save)
        y_test_pred, y_test, mseScore, rmseScore, R2Score = Predict(args, model, x_test, y_test, scaler)
        Visualize(y_test_pred, y_test, args.result_pred, "test")

    elif args.state == "iter" :
        mseL = []
        rmseL = []
        r2L = []
        model = torch.load(args.model_save)
        for i in range(args.iternum) :
            y_test_pred, _, mseScore, rmseScore, R2Score = Predict(args, model, x_test, y_test, scaler)
            mseL.append(float(mseScore.detach().numpy()))
            rmseL.append(rmseScore)
            r2L.append(R2Score)

        scoreDf = pd.DataFrame({ "MSE" : mseL,
                                "RMSE" : rmseL,
                                "R2 Score": r2L })


        print("test ",args.iternum,"번에 대한 평균값 : " )
        print(scoreDf.describe().iloc[1])

        print(scoreDf.describe())

    else :
        # viewer
        print("모델에 사용되는 데이터")
        print(resultDf.info())
        print(resultDf.head())
        print(resultDf.tail())
        print("학습에 사용되는 데이터 : ", len(x_train))
        print("         x_train.shape - ", x_train.shape)
        print("테스트에 사용되는 데이터 : ", len(x_test))
        print("         x_test.shape - ", x_test.shape)

        print("모델 구조")




if __name__ == '__main__':
    main()