import sys
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import csv




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


'''
    분단위로 하면 증가량이 너무 적음 -> 시간 단위로 이산값(증가값)을 값으로 갖는 column을 추가하되, 
    0.0이면 연산시 Nan 이 되니까 지수값 주기
'''

# # 연간 데이터 합치기
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# tPd = pd.read_csv('../data/tb_meter_log_201901.csv', encoding='utf-8', parse_dates=['updated'])





#
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

tPd = pd.read_csv('../data/tb_meter_log_2019.csv', encoding='utf-8', parse_dates=['updated'])
print(tPd.info())
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
