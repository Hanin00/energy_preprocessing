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
class FcLSTMFcBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, whatisX):
        # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(FcLSTMFcBlock, self).__init__()

        # base setting
        self.float_dtype = tf.float32
        self.activate_func = tf.nn.relu

        #sequence_length : xL, xM, xS의 분류
        self.whatisX = whatisX
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
        #if whatisX 이 1이면 Xs 그에 따른  forward 따로 그 외에는 LSTM 태움
        if self.whatisX == 1 :
            # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            # out = self.fc(h0.detach(), c0.detach())
            out = self.fc(x[:, -1, :])
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

class PredictModel(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layers, output_dim):
        super(PredictModel, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dropout_rate_ph = 0.02

        #base
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activate_func = tf.nn.relu

        self.xSlstm = FcLSTMFcBlock(input_dim, hidden_dim, num_layers, output_dim, 1)
        self.xMlstm = FcLSTMFcBlock(input_dim, hidden_dim, num_layers, output_dim, 0)
        self.xLlstm = FcLSTMFcBlock(input_dim, hidden_dim, num_layers, output_dim, 0)

#        self.xSAr =


    def forward(self, xSinput, xMinput, xLinput) :
        #todo xSinput는 xS의 input 수.. 아마..
        xSlstmRes,(hnS, cnS) = self.xSlstm(xSinput, hidden_dim, num_layers, output_dim,1)
        xMlstmRes, (hnM, cnM) = self.xMlstm(xMinput, hidden_dim, num_layers, output_dim,0)
        xLlstmRes, (hnL, cnL) = self.xLlstm(xLinput, hidden_dim, num_layers, output_dim,0)

        concatRes = torch.cat((xSlstmRes,xMlstmRes,xLlstmRes), self.input_dim)
        fc1 = self.fc(concatRes[:, -1, :])

        #todo fc의 각 값과 AR(Xs)을 concat -> horison 1 output을 얻을 수 있음. for 문으로 내보내는 부분을 원래 모델에서 살펴보기?

        # xSinput 사용


        # generate predictions
        pred_ta = tf.TensorArray(self.float_dtype, size=self.horizon)
        # shape -> [batch_size, hw, D]
        ar_input = self.X_ph[:, -self._hw:]
        for i in range(self.horizon):
            # shape -> [batch_size, D]
            nn_pred = self.nn_horizon_output_layer(output_state, 'nn_horizon' + str(i))
            ar_pred = self.ar_horizon_output_layer(ar_input, 'ar_horizon' + str(i))
            nn_pred += ar_pred

            pred_ta = pred_ta.write(i, nn_pred)


    def input_layer(self, x,):
        """
        :param x: a tensor with shape [..., input_dim]
        :param scope:
        :return: a tensor with shape [..., v]
        """
        layer_w = self.get_weights('layer_w', shape=[self.input_dim, self.output_dim])
        layer_b = self.get_bias('layer_b', shape=[self.output_dim])
        return self.activate_func(tensordot(x, layer_w) + layer_b)




    def nn_horizon_output_layer(self, x, scope='nn_horizon'):
        """ Generate a horizon prediction for the neural network part.
        :param x: [..., D]
        :param scope:
        :return: the prediction with shape [..., D]
        """
        keep_prob = 1.0 - self.dropout_rate_ph`
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            horizon_w = self.get_weights('horizon_w', shape=[self.v, self.D])
            horizon_b = self.get_bias('horizon_b', shape=[self.D])

            horizon_v = self.get_weights('horizon_v', shape=[self.D, self.D])
            horizon_v_b = self.get_bias('horizon_v_b', shape=[self.D])

            y_pred = self.activate_func(tensordot(x, horizon_w) + horizon_b)
            y_pred = tf.nn.dropout(y_pred, rate=1 - (keep_prob))

            # shape -> [..., D]
            y_pred = tensordot(y_pred, horizon_v) + horizon_v_b

        return y_pred

    def ar_horizon_output_layer(self, x, scope='ar_horizon'):
        """ Generate a horizon prediction for the auto regressive part.
        :param x: a tensor with shape [..., hw, D]
        :param scope:
        :return: the prediction with shape [..., D]
        """
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            # weights
            ar_w = self.get_weights('ar_w', shape=[self._hw, 1])
            ar_b = self.get_bias('ar_b', shape=[1])

            # shape -> [..., hw, D]
            ar_output = x * ar_w
            # [..., D]
            ar_output = tf.reduce_sum(input_tensor=ar_output, axis=-2) + ar_b

        return ar_output


    def get_rnn_cell(self,rnn_input, rnn_hid, dropout_rate):
        keep_prob = 1 - dropout_rate
        rnn = nn.RNNCell(rnn_input,rnn_hid)
        return rnn

    def get_weights(self, name, shape, collections=None):
        return tf.compat.v1.get_variable(name, shape=shape, dtype=self.float_dtype,
                               initializer=tf.compat.v1.glorot_normal_initializer(),
                               collections=collections)

    def get_bias(self, name, shape, collections=None):
        return tf.compat.v1.get_variable(name, shape=shape, dtype=self.float_dtype,
                               initializer=tf.compat.v1.constant_initializer(0.1),
                               collections=collections)
