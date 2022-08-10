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


#Skip connection
class PredictModel_before(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length, num_layers, output_dim):
        # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(PredictModel, self).__init__()

        # base setting
        self.float_dtype = tf.float32
        self.activate_func = tf.nn.relu

        #sequence_length : xL, xM, xS의 분류
        self.sequence_length = sequence_length
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #if sequence_num 이 1이면 Xs니까 그에 따른  forward 따로 그 외에는 LSTM 태움
        if self.sequence_length == 1 :
            out = self.fc
            #out = self.fc(h0.detach(), o0.detach())
            # out.size() -->
        else :
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            # Initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
        return out