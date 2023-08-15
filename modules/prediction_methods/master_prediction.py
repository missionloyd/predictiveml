import os
from modules.prediction_methods.create_predictions import create_predictions
from modules.prediction_methods.format_predictions import format_predictions
from modules.logging_methods.main import logger

def master_prediction(cli_args, winners_in_file_path, config):
    # Replace 'None' values in cli_args with corresponding values from config
    for key, value in cli_args.items():
        if value is None and key in config:
            config_value = config[key]
            if isinstance(config_value, list) and len(config_value) > 0:
                cli_args[key] = config_value[0]
            else:
                cli_args[key] = config_value

    results_header = config['results_header']
    y_column_mapping = config['y_column_mapping']
    y_column_flag = 'all'
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datelevel = cli_args['datelevel']
    time_step = str(cli_args['time_step'])
    y_pred_lists = []
    results = []

    # different then on-demand, needed to match db format
    for key, _ in y_column_mapping.items():
        predicted_key = key.replace('present', 'predicted')
        y_column_mapping[key] = predicted_key

    for building_file in config['building_file']:
        building = building_file.replace('.csv', '')
        cli_args['building_file'] = building

        for y_column in y_column_mapping:
            cli_args['y_column'] = y_column
            winners_in_file = f'{winners_in_file_path}/_winners_{building}_{y_column}_{time_step}_{datelevel}.in'

            if not os.path.exists(winners_in_file): continue

            start, end, y_pred_list, target_row = create_predictions(cli_args, startDateTime, endDateTime, winners_in_file, config)
            y_pred_lists.append((y_column, building, y_pred_list))
            len_y_pred_list = len(y_pred_list)

            results.append(format_predictions(start, end, y_pred_lists, y_column_mapping, len_y_pred_list, datelevel, time_step, target_row, results_header, y_column_flag))

    return results