import sys
import pandas as pd
import numpy as np
import pickle


#todo Guess?
# dev_location도 같이 줄 필요가 있는가?


'''
    data 통합 - 빈 데이터 프레임에 제공받은 데이터 concat
'''
pd.set_option('display.max_columns', None)

listA = ['tb_meter_log_201804','tb_meter_log_201805','tb_meter_log_201806','tb_meter_log_201807','tb_meter_log_201808']
#listA = ['tb_env_log_201902','tb_env_log_201903','tb_env_log_201904','tb_env_log_201905','tb_env_log_201906','tb_env_log_201907','tb_env_log_201908','tb_env_log_201909']



mtColumnName = ['cnt', 'dev_sid', 'dev_id', 'meter_id', 'dev_name', 'dev_location',
       'power_value', 'power_value2', 'meterType', 'gas_value', 'water_value',
       'hwater_value', 'heat_value', 'updated', 'magnitudeValue',
       'power_child', 'parent', 'ptf_value', 'pth_value', 'temp_t1',
       'temp_t2']

envLogName = ['sid', 'mac', 'meter_id', 'alias', 'temp_value', 'humidity_value',
       'dust1_value', 'dust2_value', 'dust3_value', 'light_value',
       'noise_value', 'co2_value', 'updated']

totalPd =pd.DataFrame(columns= mtColumnName)
#totalPd =pd.DataFrame(columns= envLogName)

for name in listA :
    path = './data/columnNameO/' + name + '.csv'
    # path = './data/envLog/' + name + '.csv'
    data = pd.read_csv(path, delimiter=';', parse_dates=['updated'])
    # 날짜순, meter_id 별 정렬 후 reset_index
    #pdData = data.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)
    pdData = data.reset_index(drop=True)
    totalPd = pd.concat([totalPd,pdData])
totalPd = totalPd.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)

with open("./data/mtLogTotal.pickle", "wb") as fw:
#with open("./data/envLog1902-09.pickle", "wb") as fw:
    pickle.dump(totalPd, fw)


sys.exit()
#------------------------------------------------------------

with open("./data/energyTotal.pickle", "rb") as fr:
    totalPd = pickle.load(fr)

totalPd = totalPd.replace(0.0, np.NaN)

#print(totalPd.isnull().sum())
print(totalPd.notnull().sum())



# 특징으로 사용 가능한 값 있는지 확인 - 다 0.0인데.. nan 으로 바꿔서 확인해도 될 것 같긴 함 -  NaN으로 바꿔야 함 string있자너
# for colName in columnName :
#     reqd_Index = totalPd[totalPd[colName] > 0.0].index.tolist()
#     if len(reqd_Index) != 0 :
#         print(colName + ' : ' + str(len(reqd_Index)))
#     else :
#         print(colName + " : 0")


sys.exit()


# print(totalPd.info())
# print(totalPd.head(5))
# print(totalPd.tail(5))
