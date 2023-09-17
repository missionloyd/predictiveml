import os
import pandas as pd
from datetime import datetime
import numpy as np
# from modules.logging_methods.main import logger

def saved_on_demand_prediction(cli_args, config):
    # Replace 'None' values in cli_args with corresponding values from config
    for key, value in cli_args.items():
        if value is None and key in config:
            config_value = config[key]
            if isinstance(config_value, list) and len(config_value) > 0:
                cli_args[key] = config_value[0]
            else:
                cli_args[key] = config_value

    prediction_data_path = config['prediction_data_path']
    results_header = config['results_header']
    results_file = config['results_file']
    building_file = cli_args['building_file']
    building = cli_args['building_file'].replace('.csv', '')
    y_column_mapping = config['y_column_mapping']
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datelevel = cli_args['datelevel']
    datetime_format = config['datetime_format']
    time_step = str(cli_args['time_step'])
    y_pred_lists = []
    results = []

    file_path = f'{prediction_data_path}/{building}_{results_file}'
    # logger(startDateTime + endDateTime)
    if os.path.exists(file_path):
        
        df = pd.read_csv(file_path)

        for col in df.columns:
            if 'predicted' in col:
                new_col = col.replace('predicted', 'present')
                df.rename(columns={col: y_column_mapping[new_col]}, inplace=True)

        df = df.set_index('ts')
        df.index = pd.to_datetime(df.index)

        # Filter the dataframe to include data within the startDateTime and endDateTime

        if startDateTime and endDateTime:
            # Convert startDateTime and endDateTime to datetime objects
            start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
            end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
            df = df.loc[(df.index >= start_datetime_obj) & (df.index <= end_datetime_obj)]
        elif startDateTime:
            # Convert startDateTime to datetime object
            start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
            df = df.loc[df.index >= start_datetime_obj]
        elif endDateTime:
            # Convert endDateTime to datetime object
            end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
            df = df.loc[df.index <= end_datetime_obj]

        df = df.reset_index()
        df['building_file'] = building_file
        df['water'] = None

        df.rename(columns={'ts': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df.replace(to_replace=[np.nan, "None"], value=None, inplace=True)        

        columns_to_keep = ['timestamp', 'bldgname', 'building_file'] + list(y_column_mapping.values())

        # Rename columns based on y_column_mapping
        df.rename(columns=y_column_mapping, inplace=True)

        # Drop columns except the ones to keep
        df = df[columns_to_keep]

        results = df

    return results