import pickle, csv, os
import pandas as pd
from modules.logging_methods.main import logger

# Load the pickled model or preprocessed query
def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Preprocess the query
def setup_prediction(cli_args, winners_in_file_path):

    # Create a dictionary or DataFrame with the preprocessed query
    building_file = cli_args['building_file'].replace('.csv', '')
    y_column = cli_args['y_column']
    startDate = cli_args['startDate']
    endDate = cli_args['endDate']
    datelevel = cli_args['datelevel']
    time_step = str(cli_args['time_step'])
    table = cli_args['table']

    # Load the CSV file
    with open(winners_in_file_path, mode='r') as results_file:
        csv_reader = csv.reader(results_file)
        
        # Find the indices of the 'bldgname', 'model_file', and 'model_data_path' columns
        headers = next(csv_reader)
        building_file_index = headers.index('building_file')
        y_column_index = headers.index('y_column')
        time_step_index = headers.index('time_step')
        datelevel_index = headers.index('datelevel')
        model_file_index = headers.index('model_file')
        model_data_path_index = headers.index('model_data_path')
        
        # Find the row with the specific bldgname
        target_row = None
        for row in csv_reader:
            if row[building_file_index] == building_file + '.csv' and row[y_column_index] == y_column and row[time_step_index] == time_step and row[datelevel_index] == datelevel:
                target_row = row
                break

    # Access the values of 'model_file' and 'model_data_path' columns in the retrieved row
    if target_row is not None:
        model_file = target_row[model_file_index]
        model_data_path = target_row[model_data_path_index]

        # Load the model
        model = load_object(model_file)

        # Load the preprocessed query
        model_data = load_object(model_data_path)

        return model, model_data, target_row
    else:
        logger(f"No row found with building_file '{building_file}.csv', y_column '{y_column}', time_step '{time_step}', datelevel '{datelevel}'.")
        return []