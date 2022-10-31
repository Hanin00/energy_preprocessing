import argparse

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.set_defaults(order=(2, 1, 2),
                            freq='D',
                            path = 'Final/Energy_pv/data/new_total_pv_0831.csv',
                            pred_s = "2021-09-01",
                            pred_e = "2021-09-30",
                            model_save = 'Final/Energy_pv/model/ARIMA_1031_fit.pkl',
                            date_column = 'updated')    # True

    # return enc_parser.parse_args(arg_str)