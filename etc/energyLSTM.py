# ref) https://eunhye-zz.tistory.com/8#google_vignette

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
import torch.nn as nn
import torch
import torch.optim as optim
import pickle
import sys

device = torch.device('cpu')

'''
    이전 7일 간의 데이터를 기반으로 다음 날의 종가를 예측함 Sequence = 7, output dimension = 1
    예측하기 위해 사용하는 데이터는 시가, 종가 등 총 5개의 column(Input demension)

'''
pd.set_option('display.max_columns', None)

''' 학습/테스트 데이터 분할 '''
# 2년 간의 데이터
# with open("./data/envLog1902-09.pickle", "rb") as fr:
#     totalPd = pickle.load(fr)


# df = pd.read_csv('./data/target102.csv', encoding='utf-8',parse_dates=['updated'])
# df = df.sort_index(ascending=False)
# #df = df[['power_value','gas_value', 'water_value']]
# df = df['power_value']
#
#


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#데이터 불러오기
df = pd.read_csv('./data/new_total_pv_0831.csv', parse_dates=['updated'], encoding ='utf-8', )
# df = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding ='utf-8', )
#df = pd.read_csv('../data/target102_3H_diff.csv', parse_dates=['updated'], encoding ='utf-8', )
df.set_index('updated', inplace=True)

#결측치 있어서 보간 필요(index를 datatime으로 해서 그런지는 모름 이유 파악 X)
#df_intp_linear = df.interpolate()
#df['power_value'] = df_intp_linear[['power_value', 'pw_diff']]
#df = df[['power_value', 'pw_diff']]
df = df[['power_value']]

## 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
seq_length = 143  # 하루에 10분 단위로 기록 할 때,  10분 단위 데이터 수는 159469개, 하루 단위 데이터 수는 1108개여서 하루 평균 143개를 기록한다고 가정
batch = 100

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
df = df.sort_index(ascending=False)
df = df.replace(np.nan, 0.00001 )
df = df.replace(0.0, 0.00001)
# 증가값이  0.0인 경우 ep


''' 
    데이터 셋 생성 및 tensor 형태로 변환
    파이토치에서는 3D 텐서의 입력을 받으므로 torch.FloatTensor를 사용해 np.array -> tensor 형태로 변경
'''

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
df = df[::-1]
train_size = int(len(df)*0.7)
train_set = df[0:train_size]
test_set = df[train_size-seq_length:]



''' 데이터 스케일링 '''
# 사용되는 설명 변수들의 크기가 서로 다르므로 각 컬럼을 0-1 사이의 값으로 스케일링

# Input scale - StandardScaler : 표준화
scaler_x = MinMaxScaler()  # 최대/최소 값이 각각 1,0이 되도록 스케일링
scaler_x.fit(train_set.iloc[:, :-1])

train_set.iloc[:, :-1] = scaler_x.transform(train_set.copy().iloc[:, :-1])
test_set.iloc[:, :-1] = scaler_x.transform(test_set.copy().iloc[:, :-1])

# Output scale
scaler_y = MinMaxScaler()
scaler_y.fit(train_set.iloc[:, [-1]])

train_set.iloc[:, -1] = scaler_y.transform(train_set.copy().iloc[:, [-1]])
test_set.iloc[:, -1] = scaler_y.transform(test_set.copy().iloc[:, [-1]])

''' 
    데이터 셋 생성 및 tensor 형태로 변환
    파이토치에서는 3D 텐서의 입력을 받으므로 torch.FloatTensor를 사용해 np.array -> tensor 형태로 변경
'''

# 데이터셋 생성 함수
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(np.array(train_set), seq_length)
testX, testY = build_dataset(np.array(test_set), seq_length)

## tensor set
X_train = torch.from_numpy(trainX).float()
y_train = torch.from_numpy(trainY).float()

X_test = torch.from_numpy(testX).float()
y_test = torch.from_numpy(testY).float()


# 텐서 형태로 데이터 정의
dataset = TensorDataset(X_train, y_train)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=True,
                        drop_last=True)

# 설정값
data_dim = 2 #특징값이 2개..(power_value, pw_value)
hidden_dim = 10
output_dim = 1
num_layers = 2
learning_rate = 0.01
nb_epochs = 100


class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def train_model(model, train_df, num_epochs=None, lr=None, verbose=10, patience=10):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    nb_epochs = num_epochs

    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)

        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples

            # seq별 hidden state reset
            model.reset_hidden_state()

            # H(x) 계산
            outputs = model(x_train)

            # cost 계산
            loss = criterion(outputs, y_train)

            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch

        train_hist[epoch] = avg_cost

        if epoch % verbose == 0:
            #print('Epoch:', '%010d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            print('Epoch : ', epoch, ' train loss : ', float(avg_cost))

        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if train_hist[epoch - patience] < train_hist[epoch]:
                print('\n Early Stopping')
                break
    return model.eval(), train_hist

# 모델 학습
#data_dim : input_dim,
net = Net(data_dim, hidden_dim, seq_length, output_dim, num_layers).to(device)
model, train_hist = train_model(net, dataloader, num_epochs = nb_epochs, lr = learning_rate, verbose = 20, patience = 10)

# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.plot(train_hist, label="Training loss")
plt.legend()
plt.show()

# Model Save & Load

# 모델 저장
# PATH = "./Timeseries_LSTM_data-02-stock_daily_.pth"
#torch.save(model.state_dict(), PATH)

# 불러오기
# model = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
# model.load_state_dict(torch.load(PATH), strict=False)
# model.eval()

#Evalauation
# 예측 테스트
with torch.no_grad():
    pred = []
    for pr in range(len(X_test)):

        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(X_test[pr], 0))
        predicted = torch.flatten(predicted).item()
        pred.append(predicted)

    # INVERSE
    pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
    testY_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1))

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

print('MAE SCORE : ', MAE(pred_inverse, testY_inverse))

## Test Fitting
fig = plt.figure(figsize=(8,3))
plt.plot(np.arange(len(pred_inverse)), pred_inverse, label = 'pred')
plt.plot(np.arange(len(testY_inverse)), testY_inverse, label = 'true')
plt.title("Loss plot")
plt.show()

