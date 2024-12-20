from datetime import datetime
import pandas as pd
from modules.imputation_methods.main import imputation
from modules.logging_methods.main import logger
import pickle, os
import numpy as np

def preprocessing(args, config):
    updated_arguments = []

    building_file = args['building_file']
    y_column = args['y_column']
    imputation_method = args['imputation_method']

    header = config['header']
    data_path = config['data_path']
    imp_path = config['imp_path']
    exclude_column = config['exclude_column']
    save_preprocessed_files = config['save_preprocessed_files']
    train_test_split = config['train_test_split']
    train_ratio_threshold = config['train_ratio_threshold']
    test_ratio_threshold = config['test_ratio_threshold']
    exclude_file = config['exclude_file']
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datetime_format = config['datetime_format']

    model_data_path = ''
    df = pd.read_csv(f'{data_path}/{building_file}')
    
    # Convert the data into a Pandas dataframe
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset=['bldgname', 'ts'])
    df = df.sort_values(['bldgname', 'ts'])
    
    # df = df.set_index('ts')

    # # Filter the dataframe to include data within the startDateTime and endDateTime
    # if startDateTime and endDateTime:
    #     # Convert startDateTime and endDateTime to datetime objects
    #     start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
    #     end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
    #     df = df.loc[(df.index >= start_datetime_obj) & (df.index <= end_datetime_obj)]
    # elif startDateTime:
    #     # Convert startDateTime to datetime object
    #     start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
    #     df = df.loc[df.index >= start_datetime_obj]
    # elif endDateTime:
    #     # Convert endDateTime to datetime object
    #     end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
    #     df = df.loc[df.index <= end_datetime_obj]

    # df = df.reset_index()

    # Group the dataframe by building name and timestamp
    groups = df.groupby('bldgname')

    # Cycle through building names if more than one building per file
    for name, group in groups:
        bldgname = name
        group = group.drop_duplicates(subset=['ts'])

        col_data = group[header]

        # Get the total number of rows in the dataframe
        total_rows = col_data.shape[0]
        split_index = int(train_test_split * total_rows)

        # Condition: The column `y_column` has enough valid values both before and after the split point,
        # and it is not excluded based on other conditions.
        if col_data[y_column].iloc[:split_index].count() >= train_ratio_threshold * split_index and \
            col_data[y_column].iloc[split_index:].count() >= test_ratio_threshold * (total_rows - split_index) and \
            building_file not in exclude_file and \
            y_column not in exclude_column:

            model_data = col_data.copy()
            model_data = model_data.rename(columns={y_column: 'y', 'ts': 'ds'})
            model_data = model_data.sort_values(['ds'])
            building_file_name = building_file.replace('.csv', '')
            model_data_path = f'{imp_path}/{building_file_name}_{y_column}_{imputation_method}'

            if save_preprocessed_files == True:
                os.makedirs(os.path.dirname(model_data_path), exist_ok=True)
                
                # Save the original values into a new column
                model_data['y'] = model_data['y'].astype(np.float32)
                model_data['y_saved'] = model_data['y']

                # Fill in missing values (preprocessing)
                model_data = imputation(model_data, imputation_method)

                with open(model_data_path, 'wb') as file:
                    pickle.dump(model_data, file)
                    
            updated_arg = {
                'building_file': building_file,
                'y_column': y_column,
                'imputation_method': imputation_method,
                'model_data_path': model_data_path,
                'bldgname': bldgname
            }
            updated_arguments.append(updated_arg)

        # else:
        #     logger(f'{building_file} // {y_column} cannot be trained. Try adjusting preprocessing/training scope in the config file.')

    if updated_arguments and model_data_path:
        return updated_arguments
    else:
        return None
