# -*- coding: utf-8 -*-
import sys
import argparse
import pandas as pd
import math, time
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from numpy import array
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import r2_score

import argparse
from .config import lstm_parse
from .MultivariateLSTM import LSTM

import warnings
warnings.filterwarnings(action="ignore")


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


def loadDataset(args) :
    df = pd.read_csv(args.path)

    # 결측치 보간
    for col in df.columns:
        df_intp_linear = df.interpolate()
        df[col] = df_intp_linear[col]

    in_cols = ['power_value', 'temp_mean', 'temp_min', 'temp_max', 'weather_warning']
    out_cols = ['power_value']  # 예측 대상

    # choose a number of time steps
    n_steps_in, n_steps_out = args.n_steps_in, args.n_steps_out  # 이전 30일 보고 하루 예측

    j = 0

    p_scaler = MinMaxScaler(feature_range=(0, 1))
    df['power_value'] = p_scaler.fit_transform(df['power_value'].values.reshape(-1,1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scaling dataset
    for col in in_cols[1:]:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


    dataset_low = np.empty((df[out_cols[j]].values.shape[0], 0))
    for i in range(len(in_cols)):
        dataset_low = np.append(dataset_low, df[in_cols[i]].values.reshape(df[in_cols[i]].values.shape[0], 1), axis=1)
    dataset_low = np.append(dataset_low, df[out_cols[j]].values.reshape(df[out_cols[j]].values.shape[0], 1), axis=1)
    scaled_data = dataset_low

    # convert into input/output
    train_set_size = args.train_set_size
    x_train, y_train = split_sequences(scaled_data[:train_set_size], n_steps_in, n_steps_out)
    x_test, y_test = split_sequences(scaled_data[train_set_size:-1, :], n_steps_in, n_steps_out)

    # make training and test sets in torch
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return x_train, x_test, y_train, y_test, scaler, p_scaler

def Train(args, model, x_train, y_train) :
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())
    loss_fn = torch.nn.MSELoss(size_average=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = np.zeros(args.num_epochs)
    for t in range(args.num_epochs):
        # Forward pass
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t != 0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model, args.model_save)

    mseScore = loss_fn(y_train, y_train_pred)
    rmseScore = math.sqrt(loss_fn(y_train, y_train_pred))
    R2Score = r2_score(y_train.detach().numpy(), y_train_pred.detach().numpy())

    print('Test MSE Score: %.8f' % (mseScore))
    print('Test RMSE Score: %.8f' % (rmseScore))
    print('Test R2 Score: %.8f' % (R2Score))

    # y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    # y_train = scaler.inverse_transform(y_train.detach().numpy())

    torch.save(model,  args.model_save)

    return model,y_train, y_train_pred,

def Predict( args,model, x_test, y_test, p_scaler ) :
    loss_fn = torch.nn.MSELoss(size_average=True)

    y_test_pred = model(x_test)
    mseScore = loss_fn(y_test, y_test_pred)
    rmseScore = math.sqrt(loss_fn(y_test, y_test_pred))
    R2Score = r2_score(y_test.detach().numpy(), y_test_pred.detach().numpy())

    print('Test MSE Score: %.8f' % (mseScore))
    print('Test RMSE Score: %.8f' % (rmseScore))
    print('Test R2 Score: %.8f' % (R2Score))


    y_test_pred = y_test_pred.detach().numpy()
    y_test = y_test.detach().numpy()


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



def main() :
    parser = argparse.ArgumentParser(description='Embedding arguments')
    lstm_parse(parser)
    args = parser.parse_args()

    x_train, x_test, y_train, y_test, scaler, p_scaler = loadDataset(args)

    if args.state == "train" :
        model = LSTM(input_dim=args.input_dim,
                     hidden_dim=args.hidden_dim,
                     output_dim=args.output_dim,
                     num_layers=args.num_layers)
        model, y_train, y_train_pred =  Train(args, model, x_train, y_train)
        Visualize(y_train_pred.detach().numpy(), y_train.detach().numpy(), args.result_train, "train")

    elif args.state == "test" :

        x_train, x_test, y_train, y_test, scaler, p_scaler = loadDataset(args)
        model = torch.load(args.model_save)
        y_test_pred, y_test, mseScore, rmseScore, R2Score  = Predict(args, model, x_test, y_test, scaler)
        Visualize(y_test_pred, y_test, args.result_pred, "test")

        y_test_pred = p_scaler.inverse_transform(y_test_pred)
        y_test = p_scaler.inverse_transform(y_test)

        yDf = pd.DataFrame({"y_pred": y_test_pred[:, 0],
                            "y_test": y_test[:, 0]
                            })

        with open(args.pred_result, 'w') as csv_file:
            yDf[-30:].to_csv(path_or_buf=csv_file)


    elif args.state == "iter" :
        mseL = []
        rmseL = []
        r2L = []
        model = torch.load(args.model_save)
        for i in range (args.iternum) :
            print(" ------ ", i," ------ ")
            y_test_pred, _, mseScore, rmseScore, R2Score = Predict(args, model, x_test, y_test, scaler)
            mseL.append(float(mseScore.detach().numpy()))
            rmseL.append(rmseScore)
            r2L.append(R2Score)
            print(" ")


        y_test_pred = p_scaler.inverse_transform(y_test_pred)
        y_test = p_scaler.inverse_transform(y_test)

        yDf = pd.DataFrame({"y_pred": y_test_pred[:, 0],
                            "y_test": y_test[:, 0]
                            })

        with open(args.pred_result, 'w') as csv_file:
            yDf[-30:].to_csv(path_or_buf=csv_file)


        scoreDf = pd.DataFrame({"MSE": mseL,
                                "RMSE": rmseL,
                                "R2 Score": r2L})

        print("test ",args.iternum,"번에 대한 평균값 : " )
        print(scoreDf.describe().iloc[1])

        # print(scoreDf.describe())

    else :
        # viewer
        print("모델에 사용되는 데이터")
        print()
        print("학습에 사용되는 데이터 : ", len(x_train))
        print("         x_train.shape - ", x_train.shape)
        print("테스트에 사용되는 데이터 : ", len(x_test))
        print("         x_test.shape - ", x_test.shape)

        print("모델 구조")
        model = torch.load(args.model_save)
        print(model)


if __name__ == '__main__':
    main()