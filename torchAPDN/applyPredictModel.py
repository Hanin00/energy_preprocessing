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

path = '../data/dailyPred/target102_10T_diff.csv'  #<- 일자별 max 값으로 들어가야 함. 누적값 말고 증가량이 들어가야 할 듯? 그런데 아예 0인 값들도 있는데 이걸 어카냔 말임... 아예 없는거 아니면 exp7 이런거 넣어야 하나?
dataXs = pd.read_csv(path, delimiter=',')
totalXs = dataXs["pw_diff"]

path = '../data/dailyPred/target102_xML.csv'  #<- 일자별 max 값으로 들어가야 함. 누적값 말고 증가량이 들어가야 할 듯? 그런데 아예 0인 값들도 있는데 이걸 어카냔 말임... 아예 없는거 아니면 exp7 이런거 넣어야 하나?
#todo : ? - window 가 30, 90 으로 들어가면 되는건가? LSTM  에서? 그러면 데이터 다시 뽑아야 할 듯 - 일자별로~
dataXml = pd.read_csv(path, delimiter=',')
totalXm = dataXml["pw_diff"]
totalXl= dataXml["month"]



# 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
xs_Seq_length = 144 # 하루에 총 몇 개의 데이터가 들어가는 지 확인해야함
xm_Seq_length = 30 # 일주일
xl_Seq_length = 90 # 한 달(28일)

batch = 100

# 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
xsDf = dataXs[::-1]
xsTrain_size = int(len(xsDf)*0.7)
xsTrain_set = Xsdf[0:xsTrain_size]
xsTest_set = Xsdf[xsTrain_size-seq_length:]

xmDf = dataXml[::-1]
xmTrain_size = int(len(xmDf)*0.7)
xmTrain_set = Xmdf[0:xmTrain_size]
xmTest_set = Xsdf[xmTrain_size-seq_length:]














