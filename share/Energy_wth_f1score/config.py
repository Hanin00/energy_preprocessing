import argparse


def lstm_parse(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.add_argument('--state', required=False, default="train", )
    enc_parser.add_argument('--iternum', required=False, default=10, )
    enc_parser.set_defaults(
        date_column='updated',
        freq='1D',
        target_name='power_value',
        look_back=28,
        input_dim=5,  # feature 개수 - power value 까지
        hidden_dim=32,
        num_layers=5, #hidden layer num
        output_dim=1,
        num_epochs=20,
        lr=0.01,

        n_steps_in = 30, #예측에 사용되는 이전 데이터 수
        n_steps_out = 1, #예측 범위

        train_set_size = 900,

        path='Energy_wth_f1score/data/pv_weather.csv',
        model_save='Energy_wth_f1score/model/model_e2000.pt',
        pred_result='Energy_wth_f1score/output/lstm_predict.csv',
        result_train='Energy_wth_f1score/output/lstm_train.png',
        result_pred='Energy_wth_f1score/output/lstm_predict.png',
    )  # True

# python -m Final.Energy_pv.Train