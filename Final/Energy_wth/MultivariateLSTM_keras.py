# ref : https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

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
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))  # input_shape <-t rainX.shape[1],trainX.shape[2]
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)

    return grid_model

#todo - 1. preprocessing - split & norm- on data
#data load and split
# train.csv : 2001.01.25~2021.09.29
# test.csv : 2021.09.30~2021.11.10
df=pd.read_csv("./data/train.csv",parse_dates=["Date"],index_col=[0]) #(5203, 5)
test_split=round(len(df)*0.20)
df_for_training=df[:-1041] #(4162, 5)
df_for_testing=df[-1041:] #(1041, 5)

#normalize
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
# print(df_for_training_scaled)

trainX,trainY=createXY(df_for_training_scaled,30) #
testX,testY=createXY(df_for_testing_scaled,30)


#todo - 2. training
# gridSearchCV :
# https://wikidocs.net/87220
grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
# # parameters = {'batch_size' : [16,20],
# #               'epochs' : [8,10],
# #               'optimizer' : ['adam','Adadelta'] }   #리스트로 각 파라미터의 범위를 정해서 하이퍼파라미터의 최적값을 반환받음
#
parameters = {'batch_size' : [20],
              'epochs' : [30],
              'optimizer' : ['adam'] }   #리스트로 각 파라미터의 범위를 정해서 하이퍼파라미터의 최적값을 반환받음
grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search = grid_search.fit(trainX,trainY)

print(grid_search.best_params_) # 각 하이퍼 파라미터의 최적값을 찾아줌

#save model - bset_estimator
my_model=grid_search.best_estimator_.model

#todo - 3. test - make prediction
prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

# scaler.inverse_transform(prediction) # <- while scaling test data, there were 5column for each row. but predict data has only 1 column
# So, Have to change shape.
# And, the 1st column after inverse transform is predicted value

prediction_copies_array = np.repeat(prediction,5, axis=-1)
print(prediction_copies_array.shape)
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),5)))[:,0]

# compare predicted value with testY. testY is also scaled -> have to use inverse transform.
original_copies_array = np.repeat(testY,5, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),5)))[:,0]

print("Pred Values-- " ,pred)
print("\nOriginal Values-- " ,original)


plt.plot(original, color = 'red', label = 'Real Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



# 예측 시점의 다른 특징값들은 주어지는지? 아니면 그것들도 모두 예측해야 하는지? 모두 예측같긴 하다만... 값 하나에 대해서만 예측을 하는 건지, 미래 예측에 사용되는 feature 값들도 같이 예측을 해야 하는지?



# dataset에서 "open"을 target으로예측할 것.
df_30_days_past=df.iloc[-30:,:] #31번 값을 예측하기 위해 이전 30일의 데이터를살펴봄
print(df_30_days_past.tail()) # open 값이 모두 존재함을 확인할 수 있음

df_30_days_future=pd.read_csv("./data/test.csv",parse_dates=["Date"],index_col=[0])
#1. 예측 시 예측 대상이되는 "open"열은 test 데이터에 없으므로 open 열을 생성하고, nan 으로 바꿈
#2. 이전 30일 값과 앞으로 예측할 30일의 새 값을 연결함
print(df_30_days_future.head())
df_30_days_future["Open"]=0
print(df_30_days_future.head())

df_30_days_future = df_30_days_future[["Open","High","Low","Close","Adj Close"]]
old_scaled_array=scaler.transform(df_30_days_past)
new_scaled_array=scaler.transform(df_30_days_future)
new_scaled_df=pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:,0]=np.nan
full_df=pd.concat([pd.DataFrame(old_scaled_array),new_scaled_df]).reset_index().drop(["index"],axis=1)

print(full_df.head())

full_df_scaled_array=full_df.values
all_data=[]
time_step=30 # 데이터가 없는 시점으로부터 30일 이후 예측
for i in range(time_step,len(full_df_scaled_array)):
    data_x=[]
    data_x.append(
        full_df_scaled_array[i-time_step :i , 0:full_df_scaled_array.shape[1]])
    data_x=np.array(data_x)
    prediction=my_model.predict(data_x)
    print("prediction : ", prediction)
    all_data.append(prediction) #prediction 만 모음.
    full_df.iloc[i,0]=prediction # 예측 대상인 "open" 열이 df에서 0번에 있음


new_array=np.array(all_data)
new_array=new_array.reshape(-1,1)
prediction_copies_array = np.repeat(new_array,5, axis=-1)  # inverse 시 shape이 안맞는 문제를 해결하기 위해, #targetDim+#featureDim인 5개의 열을 만듦
y_pred_future_30_days = scaler.inverse_transform(np.reshape(prediction_copies_array,(len(new_array),5)))[:,0]
resultDf = pd.DataFrame({'result': y_pred_future_30_days.tolist()})

print(len(resultDf))
print(resultDf)
print(y_pred_future_30_days) # denormalize 한 값



# test에 사용되는 feature 값은 주어진 상태