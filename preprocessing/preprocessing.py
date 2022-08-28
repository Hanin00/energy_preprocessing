# -*- coding: utf-8 -*-


import sys
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import csv




# 전체 데이터에서 갱신되지 않는 달들을 제외하고 데이터를 생성
# 2019-09 ~ 2021-08


df1D = pd.read_csv('../data/total_pv.csv', parse_dates=['updated'], encoding='utf-8',)
df1D.set_index('updated', inplace=True)
filtered_df=df1D.loc['2019-01-01':'2021-08-31']

filtered_df.to_csv("./data/total_pv_0831.csv")
sys.exit()























# todo 지금 단일 가구에 대해서 추출해서 하고 있음. 이걸 전체 가구에 사용 하려면 지금 데이터 형태로 추출하는게 필요함
'''
    현재 데이터 형태
    - 하루 단위 데이터 
    - 10분 단위 데이터로 하루 예측을 해야 하는데 없음
    
    전체 데이터 형태
    - 가구 이름 : 
    names = ['408호', '810호', '704호', '712호', '311호', '수팰리스8층환경2', '511호', '606호', '802호', '508호', '512호', '307호', '402호', '409호', '312호', '702호', '506호', '808호', '410호', '703호', '509호', '709호', '8층환경1', '604호', '710호', '608호', '405호', '609호', '707호', '603호', '412호', '4층공용', '706호', '804호', '309호', '101호', '203호(고시텔)', '8층환경', '7층공용', '8층공용', '303호', '502호', '5층공용', '302호', '807호', '411호', '711호', '806호', '501호', '705호', '404호', '202호', '301호', '612호', '801호', '3층공용', '505호', '403호', '103호', '406호', '803호', '304호', '605호', '8층환경2', '507호', '601호', '1층공용', '805호', '306호', '607호', '503호', '701호', '102호', '수팰리스8층환경1', '510호', '504호', '611호', '6층공용', '401호', '809호', '2층공용1', '602호', '407호', '708호', '610호', '310호', '308호', '305호', '201호']
    
    수행 해야 하는 것
    - 정규화
    - 결측치 보간  
    
'''

#names로 이름 변경
names = ['408호', '810호', '704호', '712호', '311호', '수팰리스8층환경2', '511호', '606호', '802호', '508호', '512호', '307호', '402호', '409호', '312호', '702호', '506호', '808호', '410호', '703호', '509호', '709호', '8층환경1', '604호', '710호', '608호', '405호', '609호', '707호', '603호', '412호', '4층공용', '706호', '804호', '309호', '101호', '203호(고시텔)', '8층환경', '7층공용', '8층공용', '303호', '502호', '5층공용', '302호', '807호', '411호', '711호', '806호', '501호', '705호', '404호', '202호', '301호', '612호', '801호', '3층공용', '505호', '403호', '103호', '406호', '803호', '304호', '605호', '8층환경2', '507호', '601호', '1층공용', '805호', '306호', '607호', '503호', '701호', '102호', '수팰리스8층환경1', '510호', '504호', '611호', '6층공용', '401호', '809호', '2층공용1', '602호', '407호', '708호', '610호', '310호', '308호', '305호', '201호']

#todo modulize 1 - pandas filtering 해서 sort, normalize 하는 함수 생성
#todo modulize 2 - 따흐흑.. 이거 결과값도 normalize 해야 할 듯



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


'''
    분단위로 하면 증가량이 너무 적음 -> 시간 단위로 이산값(증가값)을 값으로 갖는 column을 추가하되, 
    0.0이면 연산시 Nan 이 되니까 지수값 주기
'''

#tPd = pd.read_csv('../data/tb_meter_log_2019.csv', encoding='utf-8', parse_dates=['updated'])
#tPd = pd.read_csv('./data/groupby.csv', encoding='utf-8', parse_dates=['updated'])
tPd = pd.read_csv('./data/df2.csv', encoding='utf-8', parse_dates=['updated'])



print(tPd.head(5))

# updated updated 별로 합계를 냄. 그 다음에 결측치 보정을 하면 되는 거 아님? 결측치 보정을 하게 되면 줠라 많기야 하겠지만

# grpDf = pd.DataFrame(tPd.groupby('updated')['power_value'].sum())
# grpDf.to_csv("./data/total_pv.csv")
# print(tPd.head(10))
#print(tPd['dev_name'] == '702호') -> F/T

# 호수별 필터링
name = '702호'
data = tPd[tPd['dev_name'] == name]
# 결측치 보간 및 일별 데이터로 추출

df2 = tPd.loc[tPd['dev_name'] == '702호', :]

#print(df2.head(10))
sys.exit()



print(tPd.info())
print(tPd.head(10))
print(list(set(tPd['dev_name'].tolist())))



sys.exit()



pd.set_option('display.max_rows', None)

tPd = pd.read_csv('../data/tb_meter_log_2019.csv', encoding='utf-8', parse_dates=['updated'])
#tPd = pd.read_csv('../data/target102.csv', encoding='utf-8', parse_dates=['updated'])

tPd.set_index('updated', inplace=True)
tPd = tPd.resample('M').last()
df_intp_linear = tPd.interpolate()
tPd["power_value"] = df_intp_linear[["power_value"]]
tPd["pw_diff"] = tPd["power_value"].diff()
tPd['updated'] = tPd.index
tPd['YearMonth'] = pd.to_datetime(tPd["updated"]).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))

grpdf = pd.DataFrame(tPd.groupby('YearMonth')['pw_diff'].sum())
tPd['month'] = [grpdf.loc[s][0] for s in tPd['YearMonth']]

tPd.to_csv('../data/tb_2019_1M.csv')



#일 별 데이터에 월 별, 3개월 별 max 값을 갖는 column을 갖는 pd를 xM과 xL로 사용

sys.exit()


# 시단위 데이터 집계 resample / https://rfriend.tistory.com/494
tPd.set_index('updated', inplace=True)
#tPd = tPd.resample('1H').last() #Nan 값이 많아서 3H로 변경
#tPd = tPd.resample('3H').last()
#선형 보간
df_intp_linear = tPd.interpolate()
tPd["power_value"] = df_intp_linear[["power_value"]]

# #누적값 차이를 시간대 별로 칼럼으로 갖도록, 시간대 없이 단순히 인덱스 화 시킨 데이터도
# 백분률 : .pct_change / 이산 : .diff
tPd["pw_diff"] = tPd["power_value"].diff()

#tPd = pd.read_csv('./data/target1021H_diff.csv', encoding='utf-8',parse_dates=['updated'])


tPd.to_csv('./data/target102_1M_diff.csv')

sys.exit()




## 시단위 데이터 집계 resample / https://rfriend.tistory.com/494
# tPd = tPd.updated.resample('1H').last()
# pd1D = tp.resample('1H').last()
# print(tPd.head(100))
# pd1D = tp.resample('1D').last() #하루 단위 resampling



'''
    data 통합 - 빈 데이터 프레임에 제공받은 데이터 concat
'''
# pd.set_option('display.max_columns', None)
#
# listA = ['tb_meter_log_201804','tb_meter_log_201805','tb_meter_log_201806','tb_meter_log_201807','tb_meter_log_201808']
# #listA = ['tb_env_log_201902','tb_env_log_201903','tb_env_log_201904','tb_env_log_201905','tb_env_log_201906','tb_env_log_201907','tb_env_log_201908','tb_env_log_201909']
#
# mtColumnName = ['cnt', 'dev_sid', 'dev_id', 'meter_id', 'dev_name', 'dev_location',
#        'power_value', 'power_value2', 'meterType', 'gas_value', 'water_value',
#        'hwater_value', 'heat_value', 'updated', 'magnitudeValue',
#        'power_child', 'parent', 'ptf_value', 'pth_value', 'temp_t1',
#        'temp_t2']
#
# envLogName = ['sid', 'mac', 'meter_id', 'alias', 'temp_value', 'humidity_value',
#        'dust1_value', 'dust2_value', 'dust3_value', 'light_value',
#        'noise_value', 'co2_value', 'updated']
#
# totalPd =pd.DataFrame(columns= mtColumnName)
# #totalPd =pd.DataFrame(columns= envLogName)
#
# for name in listA :
#     path = './data/columnNameO/' + name + '.csv'
#     # path = './data/envLog/' + name + '.csv'
#     data = pd.read_csv(path, delimiter=';', parse_dates=['updated'])
#     # 날짜순, meter_id 별 정렬 후 reset_index
#     #pdData = data.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)
#     pdData = data.reset_index(drop=True)
#     totalPd = pd.concat([totalPd,pdData])
# totalPd = totalPd.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)
#
# with open("./data/mtLogTotal.pickle", "wb") as fw:
# #with open("./data/envLog1902-09.pickle", "wb") as fw:
#     pickle.dump(totalPd, fw)
#
#
# sys.exit()
#------------------------------------------------------------

'''
    일별
    주별
    월별
'''
# with open("./data/envLog1902-09.pickle", "rb") as fr:
#     totalPd = pickle.load(fr)
# totalPd.to_csv('./data/envLog1902-09.csv')

# with open("./data/mtLogTotal.pickle", "rb") as fr:
#    totalPd = pickle.load(fr)
# totalPd.to_csv('./data/mtLogTotal.csv')






'''
     연간 데이터 합치기
'''
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# tPd = pd.read_csv('../data/tb_meter_log_201901.csv', encoding='utf-8', parse_dates=['updated'])

# csv1 = '../data/tb2019/tb_meter_log_201901.csv'
# csv2 = '../data/tb2019/tb_meter_log_201902.csv'
# csv3 = '../data/tb2019/tb_meter_log_201903.csv'
# csv4 = '../data/tb2019/tb_meter_log_201904.csv'
# csv5 = '../data/tb2019/tb_meter_log_201905.csv'
# csv6 = '../data/tb2019/tb_meter_log_201906.csv'
# csv7 = '../data/tb2019/tb_meter_log_201907.csv'
# csv8 = '../data/tb2019/tb_meter_log_201908.csv'
# csv9 = '../data/tb2019/tb_meter_log_201909.csv'
# csv10 = '../data/tb2019/tb_meter_log_201910.csv'
# csv11 = '../data/tb2019/tb_meter_log_201911.csv'
# csv12 = '../data/tb2019/tb_meter_log_201912.csv'
#
#
# print("*** Merging multiple csv files into a single pandas dataframe ***")

# # merge files
# dataFrame = pd.concat(
#    map(pd.read_csv, [csv1,csv2,csv3,csv4,csv5,csv6,csv7,csv8,csv9,csv10,csv11,csv12]), ignore_index=True)
# print(dataFrame.info())
# print(dataFrame.head(5))
# print(dataFrame.tail(5))

# dataFrame.to_csv('../data/tb_meter_log_2019.csv')
# sys.exit()