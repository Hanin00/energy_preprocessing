#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 16:55
@desc:
"""

import numpy as np
from abc import abstractmethod

'''
    Scaler
    특성의 범위(또는 분포)를 같게 만듦
    학습에 사용될 데이터를 단위 통일화 시키는 작업
    ex) Standardization(표준화), Normalization(정규화)
    Standard Scaler : 특성돌의 평균을 0, 분산을 1로 스케일링 하는 것(정규분포로 만듦)

        
    ref) https://wooono.tistory.com/96
'''

class Scaler(object):

    @abstractmethod
    def fit(self, records):
        pass

    @abstractmethod
    def fit_scaling(self, records):
        pass

    @abstractmethod
    def scaling(self, records):
        pass

    @abstractmethod
    def inverse_scaling(self, scaled_records):
        pass


class StandZeroMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._max_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    # feature의 값을 일정한 범위 또는 규칙에 따르게 하기 위해 사용
    def scaling(self, records):
        if self._max_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return records / (self._max_val + self._epsilon)

    # 스케일링 된 결과값으로 본래 값을 구함
    def inverse_scaling(self, scaled_records):
        if self._max_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return scaled_records * (self._max_val + self._epsilon)


#feature들을 특정 범위로 스케일링, 이상치에 매우 민감, 분류보다 회귀에 유용
class MinMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = None
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._min_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=0)
        self._min_val = np.min(records, axis=0)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._min_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._min_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val


