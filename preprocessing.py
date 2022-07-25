import sys
import pandas as pd
import numpy as np
import pickle

import csv

tPd = pd.read_csv('./data/target102.csv', encoding='utf-8',parse_dates=['updated'])

print(tPd.info())
print(tPd.head(5))








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
