import pickle, csv
import pandas as pd
from modules.logging_methods.main import logger

# Load the pickled model or preprocessed query
def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Preprocess the query
def setup_prediction(args, winners_in_file_path):

    # Create a dictionary or DataFrame with the preprocessed query
    building_file = args['building_file']
    y_column = args['y_column']
    # startDate = args['startDate']
    # endDate = args['endDate ']
    # datelevel = args['datelevel']
    # table = args['table']

    # Load the CSV file
    with open(winners_in_file_path, mode='r') as results_file:
        csv_reader = csv.reader(results_file)
        
        # Find the indices of the 'bldgname', 'model_file', and 'model_data_path' columns
        headers = next(csv_reader)
        building_file_index = headers.index('building_file')
        y_column_index = headers.index('y_column')
        model_file_index = headers.index('model_file')
        model_data_path_index = headers.index('model_data_path')
        
        # Find the row with the specific bldgname
        target_row = None
        for row in csv_reader:
            if row[building_file_index] == building_file and row[y_column_index] == y_column:
                target_row = row
                break

    # Access the values of 'model_file' and 'model_data_path' columns in the retrieved row
    if target_row is not None:
        model_file = target_row[model_file_index]
        model_data_path = target_row[model_data_path_index]
        
        # logger(f"Row with bldgname '{bldgname}':")
        # logger(f"model_file: {model_file}")
        # logger(f"model_data_path: {model_data_path}")

        # Load the model
        model = load_object(model_file)

        # Load the preprocessed query
        input_data = load_object(model_data_path)

        # Make prediction
        # y_pred = model.predict(input_data

        return model, input_data, target_row
    else:
        logger(f"No row found with building_file '{building_file}' & y_column {y_column}.")
        return []