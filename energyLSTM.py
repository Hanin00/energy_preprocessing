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
df = pd.read_csv('./data/target1021D.csv',parse_dates=['updated'],  encoding = 'utf-8', )
df.set_index('updated', inplace=True)

#결측치 있어서 보간 필요(index를 datatime으로 해서 그런지는 모름 이유 파악 X)
df_intp_linear = df.interpolate()
df = df_intp_linear[["power_value"]]

## scaling
scaler = MinMaxScaler()
df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))

# 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
seq_length = 7
batch = 100

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
df = df.sort_index(ascending=False)


''' 
    데이터 셋 생성 및 tensor 형태로 변환
    파이토치에서는 3D 텐서의 입력을 받으므로 torch.FloatTensor를 사용해 np.array -> tensor 형태로 변경
'''

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

data_X, data_Y = make_dataset(df)  #(1108,  1)

train_data, train_label = data_X[:-300, ], data_Y[:-300, ]  #(788,20,1),(788, 1)
test_data, test_label = data_X[-300:, ], data_Y[-300:,]  #(300, 20, 1), (300, 1)


## tensor set
X_train = torch.from_numpy(train_data).float()
y_train = torch.from_numpy(train_label).float()

X_test = torch.from_numpy(test_data).float()
y_test = torch.from_numpy(test_label).float()


# 텐서 형태로 데이터 정의
dataset = TensorDataset(X_train, y_train)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=True,
                        drop_last=True)

# 설정값
data_dim = 1 #특징값이 3개..(power_value, gas_value, water_value)
hidden_dim = 10
output_dim = 1
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
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if train_hist[epoch - patience] < train_hist[epoch]:
                print('\n Early Stopping')

                break

    return model.eval(), train_hist

# 모델 학습
net = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
model, train_hist = train_model(net, dataloader, num_epochs = nb_epochs, lr = learning_rate, verbose = 20, patience = 10)


# epoch별 손실값
fig = plt.figure(figsize=(10, 4))
plt.plot(train_hist, label="Training loss")
plt.legend()
plt.show()


# Model Save & Load
# 모델 저장
PATH = "./Timeseries_LSTM_data-02-stock_daily_.pth"
torch.save(model.state_dict(), PATH)

# 불러오기
model = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
model.load_state_dict(torch.load(PATH), strict=False)
model.eval()

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
    pred_inverse = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
    testY_inverse = scaler.inverse_transform(y_test)

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

print('MAE SCORE : ', MAE(pred_inverse, testY_inverse))

fig = plt.figure(figsize=(8,3))
plt.plot(np.arange(len(pred_inverse)), pred_inverse, label = 'pred')
plt.plot(np.arange(len(testY_inverse)), testY_inverse, label = 'true')
plt.title("Loss plot")
plt.show()



