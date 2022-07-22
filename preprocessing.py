import sys
import pandas as pd
import numpy as np
import pickle

'''
    data 통합 - 빈 데이터 프레임에 제공받은 데이터 concat
'''
pd.set_option('display.max_columns', None)

listA = ['tb_meter_log_201804','tb_meter_log_201805','tb_meter_log_201806','tb_meter_log_201807','tb_meter_log_201808']

columnName = ['cnt', 'dev_sid', 'dev_id', 'meter_id', 'dev_name', 'dev_location',
       'power_value', 'power_value2', 'meterType', 'gas_value', 'water_value',
       'hwater_value', 'heat_value', 'updated', 'magnitudeValue',
       'power_child', 'parent', 'ptf_value', 'pth_value', 'temp_t1',
       'temp_t2']


totalPd =pd.DataFrame(columns= columnName)

for name in listA :
    path = './data/columnNameO/' + name + '.csv'
    #path = './data/columnNameX/' + name + '.csv'
    data = pd.read_csv(path, delimiter=';')
    # 날짜순, meter_id 별 정렬 후 reset_index
    #pdData = data.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)

    pdData = data.reset_index(drop=True)
    totalPd = pd.concat([totalPd,pdData])

totalPd = data.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)

with open("./data/energyTotal.pickle", "wb") as fw:
    pickle.dump(totalPd, fw)

#------------------------------------------------------------



with open("./data/energyTotal.pickle", "rb") as fr:
    totalPd = pickle.load(fr)


reqd_Index = totalPd[totalPd['gas_value'] > 1.0].index.tolist() #
print(reqd_Index)
print(totalPd.iloc[reqd_Index[0]])
print(len(reqd_Index))
sys.exit()






#reqd_Index = totalPd[totalPd['water_value'] > 1.0].index.tolist()
reqd_Index = totalPd[totalPd['gas_value'] > 1.0].index.tolist() #
#reqd_Index = totalPd[totalPd['hwater_value'] > 1.0].index.tolist() #0

print(reqd_Index)
print(totalPd.iloc[reqd_Index[0]])
print(len(reqd_Index))
sys.exit()


# print(totalPd.info())
# print(totalPd.head(5))
# print(totalPd.tail(5))
