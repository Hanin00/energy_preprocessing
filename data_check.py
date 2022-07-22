import sys
import pickle
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

#with open("./data/envLog1902-09.pickle", "rb") as fr:
with open("./data/mtLogTotal.pickle", "rb") as fr:
    totalPd = pickle.load(fr)

print(totalPd['gas_value'])
print(totalPd['water_value'])
sys.exit()

# totalPd = totalPd.replace(0.0, np.NaN)
# totalPd = totalPd.replace(0, np.nan)





# print(totalPd.notnull().sum())
# print(totalPd.info())


df = totalPd.copy()


df.set_index('updated', inplace=True)

print("일별 데이터 평균 개수 : ")
print(str((df.groupby(pd.Grouper(freq='D')).count().mean()) ))
print("주별 데이터 평균 개수 : ")
print(str((df.groupby(pd.Grouper(freq='W')).count().mean()) ))
print("월별 데이터 평균 개수 : " )
print(str((df.groupby(pd.Grouper(freq='M')).count().mean()) ))

sys.exit()



# df['day']=df.index.day
# df['week']=df.index.week
# df['month']=df.index.month
# df['year']=df.index.year
# dfin = df.groupby(['year', 'month','week','day']).size()


sys.exit()






name = 'tb_env_log_201903'
#listA = ['tb_env_log_201902','tb_env_log_201903','tb_env_log_201904','tb_env_log_201905','tb_env_log_201906','tb_env_log_201907','tb_env_log_201908','tb_env_log_201909']
listA = ['tb_meter_log_201804','tb_meter_log_201805','tb_meter_log_201806','tb_meter_log_201807','tb_meter_log_201808']
#listA = ['tb_meter_log_201805',]


for name in listA :
    path = './data/envLog/' + name + '.csv'
    data = pd.read_csv(path, delimiter=';')

    # 날짜순, meter_id 별 정렬 후 reset_index
    pdData = data.sort_values(by=['updated', 'meter_id']).reset_index(drop=True)

    print("==========="+name+"===========")
    print("info - "+name+"===========")
    print(pdData.info())
    print(".isnull().sum() - " + name + "===========")
    print(pdData.isnull().sum())
    print('\n' + "head - " + name+"===========")
    print(pdData.head())
    print('\n' + "tail - " + name+"===========")
    print(pdData.tail())








    '''
    201902 : 없음
    201903 : 2019-03-02 14:20:15 ~ 2019-03-31 23:08:09
    그 외(04-09) : 2019-03-05 13:07:59 ~ 2019-04-17 06:56:35
    결측치 없음
            row num  
    201902  0        
    201903  7992     
    201904  2184     
    201905  6704     
    201906  6818     
    201907  8472     
    201908  8564     
    201909  8632     
    

    tb_meter_log_201804 : 2018-04-20 14:06:27 ~ 2018-04-30 23:57:50  결측치 없음   419724
    tb_meter_log_201805 : 2018-05-01 00:00:50 ~ 2018-05-31 23:58:02  결측치 없음   1079782
    tb_meter_log_201806 : 2018-06-01 00:01:02 ~ 2018-06-30 23:59:03  결측치 없음   1164594
    tb_meter_log_201807 : 2018-07-01 00:02:03 ~ 2018-07-31 23:58:46  결측치 없음   1180909
    tb_meter_log_201808 : 2018-08-01 00:01:46 ~ 2018-08-05 16:14:39  결측치 없음   188638
    
    
    일자별로 몇 개씩 있는지 확인 필요
    
    
    
    
    '''