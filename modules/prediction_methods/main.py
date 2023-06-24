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
    bldgname = args['bldgname'].replace('_Data', '')
    y_column = args['y_column']
    startDate = args['startDate']
    endDate = args['endDate']
    datelevel = args['datelevel']
    table = args['table']

    # Load the CSV file
    with open(winners_in_file_path, mode='r') as results_file:
        csv_reader = csv.reader(results_file)
        
        # Find the indices of the 'bldgname', 'model_file', and 'model_data_path' columns
        headers = next(csv_reader)
        bldgname_index = headers.index('bldgname')
        y_column_index = headers.index('y_column')
        model_file_index = headers.index('model_file')
        model_data_path_index = headers.index('model_data_path')
        
        # Find the row with the specific bldgname
        target_row = None
        for row in csv_reader:
            if row[bldgname_index] == bldgname and row[y_column_index] == y_column:
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

        # frequency_mapping = {
        #     'hour': 'H',
        #     'day': 'D',
        #     'month': 'M',
        #     'year': 'Y'
        # }

        if startDate and endDate:
            # Filter based on building name and date range
            input_data = input_data[(input_data['ds'] >= startDate) & (input_data['ds'] < endDate)]

            # Group by timestamp and perform aggregation
            # input_data['ds'] = pd.to_datetime(input_data['ds'])
            # input_data['ds'] = input_data['ds'].dt.to_period(frequency_mapping[datelevel])
            # select_cols = ["present_elec_kwh", "present_htwt_mmbtuh", "present_wtr_usgal", "present_chll_tonh", "present_co2_tonh", "timestamp"]
            # input_data = input_data.groupby('timestamp')[select_cols].sum()


        return model, input_data, target_row
    else:
        logger(f"No row found with bldgname '{bldgname}' & y_column {y_column}.")
        return []