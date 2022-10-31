import argparse

def arima_parse(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.set_defaults(order=(2, 1, 2),
                            freq='D',
                            path = 'Final/Energy_pv_arima/data/power_value.csv',
                            pred_s = "2021-08-01",
                            pred_e = "2021-09-30",
                            # model_save = 'Final/Energy_pv_arima/model/ARIMA_fit.pkl',
                            model_save = 'Final/Energy_pv_arima/model/ARIMA_fit.pt',
                            date_column = 'updated',
                            pred_result='Final/Energy_pv_arima/output/arima_predict.csv',
                            result_visualize='Final/Energy_pv_arima/output/arima_predict.png',
                            )    # True

    # return enc_parser.parse_args(arg_str)