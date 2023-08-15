import pickle, csv, os
import pandas as pd
from modules.prediction_methods.create_predictions import create_predictions
from modules.prediction_methods.format_predictions import format_predictions
from modules.logging_methods.main import logger

def on_demand_prediction(cli_args, winners_in_file_path, config):
    # Replace 'None' values in cli_args with corresponding values from config
    for key, value in cli_args.items():
        if value is None and key in config:
            config_value = config[key]
            if isinstance(config_value, list) and len(config_value) > 0:
                cli_args[key] = config_value[0]
            else:
                cli_args[key] = config_value

    results_header = config['results_header']
    building = cli_args['building_file'].replace('.csv', '')
    y_column_mapping = config['y_column_mapping']
    y_column_flag = cli_args['y_column']
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datelevel = cli_args['datelevel']
    time_step = str(cli_args['time_step'])
    y_pred_lists = []
    results = []

    if cli_args['y_column'] == 'all':
        for y_column in y_column_mapping:
            cli_args['y_column'] = y_column
            winners_in_file = f'{winners_in_file_path}/_winners_{building}_{y_column}_{time_step}_{datelevel}.in'
            
            if not os.path.exists(winners_in_file): continue
            
            start, end, y_pred_list, target_row = create_predictions(cli_args, config['startDateTime'], config['endDateTime'], winners_in_file, config)
            y_pred_lists.append((y_column, building, y_pred_list))

            len_y_pred_list = len(y_pred_list)
            results = format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step, target_row, results_header, y_column_flag)
    else:
        y_column = cli_args['y_column']
        
        winners_in_file = f'{winners_in_file_path}/_winners_{building}_{y_column}_{time_step}_{datelevel}.in'

        if not os.path.exists(winners_in_file): 
            logger(f"This configuration must be trained on and saved prior to making a prediction.")
            return results
        
        start, end, y_pred_list, target_row = create_predictions(cli_args, startDateTime, endDateTime, winners_in_file, config)
        y_pred_lists.append((y_column, building, y_pred_list))

        len_y_pred_list = len(y_pred_list)
        results = format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step, target_row, results_header, y_column_flag)

    # if len(results) == 0:
    #     results = {'data': [], 'status': 'empty'}
    # elif len(results) > 0:
    #     results = {'data': results['data'] or [], 'status': 'ok'}

    return results