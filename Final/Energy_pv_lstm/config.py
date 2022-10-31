import argparse

def lstm_parse(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.add_argument('--state', required=False, default = "train" ,)
    enc_parser.set_defaults(
                            date_column = 'updated',
                            freq = '1D',
                            target_name='power_value',
                            look_back = 28,
                            input_dim = 1,
                            hidden_dim = 128,
                            num_layers = 2,
                            output_dim = 1,
                            lr = 0.01,
                            num_epochs = 2000,

                            path = 'Final/Energy_pv_lstm/data/power_value.csv',
                            model_save = 'Final/Energy_pv_lstm/model/lstm.pkl',
                            pred_result='Final/Energy_pv_lstm/output/lstm_predict.csv',
                            result_train='Final/Energy_pv_lstm/output/lstm_train.png',
                            result_pred='Final/Energy_pv_lstm/output/lstm_predict.png',
                            )    # True

#python -m Final.Energy_pv_lstm.Train