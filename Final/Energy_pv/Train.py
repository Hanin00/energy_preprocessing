import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from .config import arima_parse
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import font_manager, rc

# 10T data -> 1D data resampling & 선형 보간
def resampleFreq(args, df) :
    df.set_index(args.date_column, inplace=True)
    resultDf = df.resample(args.freq).last()
    df_intp_linear = resultDf.interpolate()
    resultDf.iloc[0] = df_intp_linear.iloc[0]
    return resultDf

def Train(args, resultDf) :
    # model = ARIMA(resultDf, order, freq='D')
    model = ARIMA(resultDf, order=args.order, freq=args.freq)
    model_fit = model.fit()
    model_fit.save(args.model_save)
    print(model_fit.summary())
    return model_fit


def Predict(args, model) :
    # preds = model_fit.predict(1, 30, typ='levels')
    preds = model.predict(args.pred_s,args.pred_e, typ='levels')
    with open(args.pred_result, 'w') as csv_file:
        preds.to_csv(path_or_buf=csv_file)

    return preds
def Visualize(pred_arima_y, test_y, path) :
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 10))
    # 모델이 예측한 가격 그래프
    plt.plot(pred_arima_y, color='gold')
    # 실제 가격 그래프
    plt.plot(test_y, color='green')

    plt.legend(['예측값', '실제값'])
    plt.title("값 비교")
    plt.show()
    plt.savefig(path)

    return


def main():
    parser = argparse.ArgumentParser(description='Embedding arguments')
    arima_parse(parser)
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)
    df1D = pd.read_csv(args.path, parse_dates=[args.date_column], encoding='utf-8', )

    resultDf = resampleFreq(args, df1D)  # 일 단위 데이터로 변환 및 결측치 선형 보간
    model_fit = Train(args, resultDf)
    pred = Predict(args, model_fit)
    Visualize(pred, resultDf[-31:], args.result_visualize)

    print(pred)


if __name__ == '__main__':
    main()
