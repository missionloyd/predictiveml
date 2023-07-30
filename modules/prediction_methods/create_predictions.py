from modules.prediction_methods.predict_y_column import predict_y_column
from modules.prediction_methods.setup_prediction import setup_prediction
from modules.utils.save_args import generate_arg

def create_predictions(cli_args, startDateTime, endDateTime, winners_in_file_path, config):
    results_header = config['results_header']
    datelevel = cli_args['datelevel']
    
    model, model_data, target_row = setup_prediction(cli_args, winners_in_file_path)
    results_dict = {key: value for key, value in zip(results_header, target_row)}
    args = generate_arg(results_dict)
    y_pred_list, start, end = predict_y_column(args, startDateTime, endDateTime, config, model_data, model, datelevel)

    return start, end, y_pred_list, target_row