import argparse

def lstm_parse(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.add_argument('--state', required=False, default = "train" ,)
    enc_parser.add_argument('--iternum', required=False, default=10, )
    enc_parser.set_defaults(
                            date_column = 'updated',
                            freq = '1D',
                            target_name='power_value',
                            look_back = 29,
                            input_dim = 1,
                            hidden_dim = 128,
                            num_layers = 2,
                            output_dim = 1,
                            lr = 0.01,
                            num_epochs = 2000,

                            train_set_size=900,

                            path = 'Energy_pv/data/power_value.csv',
                            model_save = 'Energy_pv/model/model_e2000.pt',
                            pred_result='Energy_pv/output/lstm_predict.csv',
                            result_train='Energy_pv/output/lstm_train.png',
                            result_pred='Energy_pv/output/lstm_predict.png',
                            )    # True

#python -m Final.Energy_pv.Train