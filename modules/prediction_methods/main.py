import pickle, csv, os
import pandas as pd
from modules.prediction_methods.create_predictions import create_predictions
from modules.prediction_methods.format_predictions import format_predictions
from modules.logging_methods.main import logger

def predict(cli_args, winners_in_file_path, config):
    y_column_mapping = config['y_column_mapping']
    datelevel = cli_args['datelevel']
    building_file = cli_args['building_file'].replace('.csv', '')
    startDate = cli_args['startDate']
    endDate = cli_args['endDate']
    datelevel = cli_args['datelevel']
    time_step = str(cli_args['time_step'])
    y_pred_lists = []
    results = []

    if cli_args['y_column'] == 'all':
        for y_column in y_column_mapping:
            cli_args['y_column'] = y_column
            winners_in_file = f'{winners_in_file_path}/_winners_{building_file}_{y_column}_{time_step}_{datelevel}.in'
            
            if not os.path.exists(winners_in_file): continue
            
            start, end, y_pred_list = create_predictions(cli_args, winners_in_file, config)
            y_pred_lists.append((y_column, y_pred_list))

            len_y_pred_list = len(y_pred_list)
            results = format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step)
    else:
        winners_in_file = f'{winners_in_file_path}/_winners.in'

        if not os.path.exists(winners_in_file): 
            logger(f"This configuration must be trained on prior to making a prediction.")
            return results
        
        start, end, y_pred_list = create_predictions(cli_args, winners_in_file, config)
        y_column = cli_args['y_column']
        y_pred_lists.append((y_column, y_pred_list))

        len_y_pred_list = len(y_pred_list)
        results = format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step)

    return results