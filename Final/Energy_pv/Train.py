# ref) https://wikidocs.net/50949
import sys
import argparse
import pandas as pd
import statsmodels.api as sm
from .config import parse_encoder
from statsmodels.tsa.arima.model import ARIMA

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
    return preds
def Visualize() :
    return


def main():
    parser = argparse.ArgumentParser(description='Embedding arguments')
    # utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    pd.set_option('display.max_columns', None)

    df1D = pd.read_csv(args.path, parse_dates=[args.date_column], encoding='utf-8', )

    resultDf = resampleFreq(args, df1D)  # 일 단위 데이터로 변환 및 결측치 선형 보간
    model_fit = Train(args, resultDf)
    pred = Predict(args, model_fit)
    print(pred)


if __name__ == '__main__':
    main()
