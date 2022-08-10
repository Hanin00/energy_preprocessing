#ref)https://github.com/Hinterhalter/Post-COVID-19_modeling/blob/master/COVID19_Timeseries_LSTM.ipynb

import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더


import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

import sys
#ref) https://rfriend.tistory.com/494 - resample
import pandas as pd
import sys
import os
import numpy as np
import matplotlib




matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False

pd.set_option('display.max_columns', None)
#dev_id, dev_name, power_value, updated

tp = pd.read_csv('../data/tb_2019_1M.csv', parse_dates=['updated'], encoding ='utf-8', )
newTp = tp.set_index('updated')

power_valuePlt = newTp[['power_value',]]
pw_diffPlt = newTp[['pw_diff']]

power_valuePlt['2019-01':'2019-12'].plot(title = '월별 누적 전력 소모량')
pw_diffPlt['2019-01':'2019-12'].plot(title = '월별 전력 증가량')


# power_valuePlt['2019':'2022'].plot(title = 'power_value')
# pw_diffPlt['2019':'2022'].plot(title = 'pw_diff')
plt.show()

# tp = pd.read_csv('../data/target102_1D_diff.csv', parse_dates=['updated'], encoding ='utf-8', )
# print(tp.info())


sys.exit()
tp = pd.read_csv('../data/dailyPred/target102.csv', parse_dates=['updated'], encoding ='utf-8', )
tp.set_index('updated', inplace=True)

pd1D = tp.resample('1D').last()
pd1D.to_csv('./data/target1021D.csv',sep=',')

sys.exit()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

pd.set_option('display.max_columns', None)
#데이터 불러오기
df = pd.read_csv('../data/dailyPred/target102.csv', parse_dates=['updated'], encoding ='utf-8', )
df.set_index('updated', inplace=True)
print(df.head(5))

#
#
#
# #lag_col = list(df.columns)
# lag_col = df['power_value'].tolist()
#
# lag_amount = 1
#
# #lag_amount = 3
#
# for col in lag_col:
#     for i in range(lag_amount):
#         df['{0}_lag{1}'.format(col, i + 1)] = df['{}'.format(col)].shift(i + 1)

df.dropna(inplace=True)

print(df.head())

X_cols = list(df.columns)



# 테스트 데이터 수
test_data_size = 14
# X변수들과 y변수 구분
X = df[X_cols]
y = df['target']


# MinMaxScaler을 통한 스케일링
scaler = MinMaxScaler()
# X scaler 생성
Xscaler = scaler.fit(X)
# Y scaler 생성
yscaler = scaler.fit(y.values.reshape(-1,1))

# 스케일링 적용
X = Xscaler.transform(X)
y = yscaler.transform(y.values.reshape(-1,1))

# Train, Test set split
X_train, X_test = X[:-test_data_size], X[-test_data_size:]
y_train, y_test = y[:-test_data_size].flatten(), y[-test_data_size:].flatten()

print("train set : ", X_train.shape)
print("test set : ", X_test.shape)