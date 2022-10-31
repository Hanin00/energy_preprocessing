# ref : https://www.kaggle.com/code/imegirin/multivariate-time-series-modeling-with-pytorch/notebook
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

import warnings
warnings.filterwarnings(action='ignore')

#하루 예측 후 그 다음값들은 다른 feature들고 같이 들어가는지 확인. 그리고 powervalue 가 맞는지.
#지금 들어가는 꼴을 보아하니 아닌 것 같음...
#값 normalize도 안되어 있음. 해당 부분 수정해야함. 근데 어디에? pv_ 만? 온도도 해야하나?


#todo - 이 모델 사용하려면 예측값의 범위를 time index 로 갖는 dataframe을 만들어야 함



'''
    createXY() : read each steps properly
        n_past : #step we will look in the past to predict the next target value
            -> if n_past = 30, predict 31st target value
        
'''
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]]) # 맨 마지막 column인 target을 제외한 모든 feature 고려
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)


# print("trainX Shape-- ",trainX.shape)
# print("trainY Shape-- ",trainY.shape)
# print("trainX[0]-- \n",trainX[0])
# print("trainY[0]-- ",trainY[0])

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,6)))  # input_shape <-trainX.shape[1],trainX.shape[2]
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)

    return grid_model



#todo - 1. preprocessing - split & norm- on data
#data load and split
# train.csv : 2001.01.25~2021.09.29
# test.csv : 2021.09.30~2021.11.10
# df=pd.read_csv("./data/train.csv",parse_dates=["Date"],index_col=[0]) #(5203, 5)

df = pd.read_csv('./data/result_pv_1031_2.csv',
                 index_col='updated',)  #5203

time_shift = 28

target_data = df['power_value'].shift(time_shift)
data = df.iloc[:-time_shift] #5196

#spilt
# test_head = data.index[int(0.8*len(data))] # 2015-07-31
test_head = data.index[-30] # 2015-07-31
print("test_head : ",test_head)
df_train = df.loc[:test_head,:] #4157
df_test = df.loc[test_head:,:]
target_train = target_data.loc[:test_head]
target_test = target_data.loc[test_head:]

cols_means = {}
for col in df.columns:
    cols_means[col] = df_train[col].mean()
    df_train[col] = df_train[col].fillna(value=cols_means[col])
    df_test[col] = df_test[col].fillna(value=cols_means[col])

print("df_train.tail() : ",df_train.tail())
print("df_test.head() : ", df_test.head())
print("df_test.tail() : ",df_test.tail())


#dataset에 step(*sequence_length) 적용
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=6):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)  #
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

features = ['temp_mean','temp_min','temp_max','humidity_value_avg','humidity_value_min',"weather_warning"]
target = 'power_value'  #예측 대상
torch.manual_seed(42) # 초기 가중치 설정에 사용되는 random seed를 고정함

'''
#SequenceDataset(), loader 사용 예제
train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
# X1, y1 = train_dataset[0] #과거 7일은 참고할 값이 없으므로 비는 숫자만큼 반복되어 들어감 ex) 첫번째 날인 경우 첫번째 값이 7번, 두 번째 날인 경우 첫번째 값이 6번

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
X, y = next(iter(train_loader))
# print(X.shape) # torch.Size([3, 7, 4])  batch_size, seq_length, #feature
'''



batch_size = 4
sequence_length = 28 #이전 28일을 참고해 예측
#mk train_dataset
train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
#mk test_dataset
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))
print("y : ", y)

sys.exit()



def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


# todo model
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


# Model Train
# learning_rate = 5e-4
learning_rate =  0.01
num_hidden_units = 8

model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

for ix_epoch in range(10):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_model(test_loader, model, loss_function)




train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_eval_loader, model).numpy()
df_test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

# for c in df_out.columns:
#     df_out[c] = df_out[c] * target_stdev + target_mean
# print(df_out)
# print(df_out.tail())
# print(df_test.head())
# print(df_test)

print(df_test[['power_value', ystar_col]])