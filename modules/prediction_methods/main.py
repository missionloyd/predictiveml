import pickle, csv
import pandas as pd
from modules.prediction_methods.create_predictions import create_predictions
from modules.prediction_methods.format_predictions import format_predictions

def predict(cli_args, winners_in_file_path, config):
    y_column_mapping = config['y_column_mapping']
    datelevel = cli_args['datelevel']
    y_pred_lists = list()

    if cli_args['y_column'] == 'all':
        for y_column in y_column_mapping:
            cli_args['y_column'] = y_column
            start, y_pred_list = create_predictions(cli_args, winners_in_file_path, config)
            y_pred_lists.append((y_column, y_pred_list))
    else:
        start, y_pred_list = create_predictions(cli_args, winners_in_file_path, config)
        y_pred_lists.append((y_column, y_pred_list))

    len_y_pred_list = len(y_pred_list)
    results = format_predictions(start, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel)

    return results