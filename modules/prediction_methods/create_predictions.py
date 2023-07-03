from modules.prediction_methods.predict_y_column import predict_y_column
from modules.prediction_methods.setup_prediction import setup_prediction
from modules.utils.save_args import generate_arg

def create_predictions(cli_args, winners_in_file_path, config):
    results_header = config['results_header']
    datelevel = cli_args['datelevel']
    
    model, model_data, target_row = setup_prediction(cli_args, winners_in_file_path)
    results_dict = {key: value for key, value in zip(results_header, target_row)}
    args = generate_arg(results_dict, config)
    y_pred_list = predict_y_column(args, model_data, model, datelevel)
    start = model_data['ds'].iloc[-1]

    return start, y_pred_list