# 분석에 필요한 라이브러리 호출

import pandas as pd # 판다스 호출
import numpy as np  # 넘파이 호출
import statsmodels.api as sm # statsmodels 호출
import seaborn as sns # 그래프를 그리기위한 Seaborn 호출
from statsmodels.tsa.seasonal import seasonal_decompose # 데이터 필터 라이러리 호출
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA



pd.set_option('display.max_columns', None)
# 데이터 불러오기
df = pd.read_csv('../data/dailyPred/target102_10T_diff.csv', parse_dates=['updated'], encoding='utf-8', )
df.set_index('updated', inplace=True)

scaler = MinMaxScaler()
df["power_value"] = scaler.fit_transform(df["power_value"].values.reshape(-1, 1))
df = df["power_value"]
df = df[::-1]

# 최대 전력 수요 가시화
sns.set(rc={'figure.figsize':(25,10)})
sns.lineplot(x=df.index , y=df)



'''
AR이 몇번째 과거까지를 바라보는지에 대한 파라미터,    144 - 하루 전 
차분(Defference)에 대한 파라미터,                72 - 반나절 전
MA가 몇 번째 과거까지를 바라보는지에 대한 파라미터    144 - 하루 전

차분이란 현재 상태의 변수에서 바로 전 상태의 변수를 빼주는 것을 의미하며, 시계열 데이터의 불규칙성을 조금이나마 보정해주는 역할
또한 앞서 말한 ARIMA 모델의 경향성을 의미

'''
#model = ARIMA(df, order=(0,2,1))
#model_fit = model.fit(trend='nc',full_output=True, disp=1)

df_1920 = df['2019-01':'2020-12']


model = ARIMA(df_1920, order=(2,1,2))
model_fit = model.fit(trend='c',full_output=True, disp=1)
fore = model_fit.forecast(steps=144*27,alpha=0.05)
fore_df = pd.DataFrame()
result_df = df['2021-07':'2021-08']
fore_df['Real'] = result_df
fore_df.reset_index(drop=False)
fore_df['ARIMA'] = fore[0]
temp_df = df['2020-07-01':'2020-07-27'].reset_index()
fore_df.reset_index(inplace=True)
fore_df['20year'] = temp_df*1.05