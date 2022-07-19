import pandas as pd
import numpy as np

#data1902 = pd.read_csv('./data/tb_env_log_201902.csv', delimiter=';')
# data1903 = pd.read_csv('./data/tb_env_log_201903.csv', delimiter=';')
# data1904 = pd.read_csv('./data/tb_env_log_201904.csv', delimiter=';')
# data1905 = pd.read_csv('./data/tb_env_log_201905.csv', delimiter=';')
# data1906 = pd.read_csv('./data/tb_env_log_201906.csv', delimiter=';')
# data1907 = pd.read_csv('./data/tb_env_log_201907.csv', delimiter=';')
# data1908 = pd.read_csv('./data/tb_env_log_201908.csv', delimiter=';')
# data1909 = pd.read_csv('./data/tb_env_log_201909.csv', delimiter=';')

#pd.set_option('display.max_columns', None)


name = 'tb_env_log_201903'

listA = ['tb_env_log_201902','tb_env_log_201903','tb_env_log_201904','tb_env_log_201905','tb_env_log_201906','tb_env_log_201907','tb_env_log_201908','tb_env_log_201909']
#listA = ['tb_meter_log_201804']

for name in listA :
    path = './data/' + name + '.csv'
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
    
    tb_meter_log_201804 : 2018-04-20 14:06:27 ~ 2018-04-30 23:57:50 
    결측치 없음
    
    '''