1.테스트 프로그램 사용순서
$ docker run --user root -it --gpus all -v /:/work_space wedrive/energy_predict:1 /bin/bash    # 도커 컨테이너 생성
$ cd /work_space/home/dblab/energy/EnergyPred                            # 작업 디렉토리 변경
$ pip install -r "requirements.txt"                                      # 필요 모듈 설치
$ python DailyPredict.py                             # test file 실행 - 일 별 예측
$ python MonthlyPredict.py                             # test file 실행 - 월 별 예측


2.결과물
데이터 비교 그래프 :  
- 일 별 예측 : /home/dblab/energy/EnergyPred/output/DailyPredict.png
- 월 별 예측 : /home/dblab/energy/EnergyPred/output/MonthlyPredict.png

3.테스트 프로그램 설명
> 사용 모델   : 
    LSTM : /home/dblab/energy/EnergyPred/model/LSTM_model.pt
    ARIMA : /home/dblab/energy/EnergyPred/model/arima_model_fit.pt

> 사용 데이터 : /home/dblab/energy/EnergyPred/data/new_total_pv_0831.csv