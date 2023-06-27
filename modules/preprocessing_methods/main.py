import pandas as pd
from modules.imputation_methods.main import imputation
from modules.logging_methods.main import logger
import pickle

def preprocessing(args):
    updated_arguments = []

    building_file = args['building_file']
    y_column = args['y_column']
    imputation_method = args['imputation_method']
    header = args['header']
    data_path = args['data_path']
    tmp_path = args['tmp_path']
    min_number_of_days = args['min_number_of_days']
    exclude_column = args['exclude_column']
    save_preprocessed_file = args['save_preprocessed_file']
    split_rate = args['split_rate']
    exclude_file = args['exclude_file']

    logger(args)

    model_data_path = ''

    df = pd.read_csv(f'{data_path}/{building_file}')

    # Convert the data into a Pandas dataframe
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset=['bldgname', 'ts'])
    df = df.sort_values(['bldgname', 'ts'])

    # Group the dataframe by building name and timestamp
    groups = df.groupby('bldgname')
    df = df.set_index('ts')

    # Cycle through building names if more than one building per file
    for name, group in groups:
        bldgname = name
        group = group.drop_duplicates(subset=['ts'])

        col_data = group[header]

        # Get the total number of rows in the dataframe
        total_rows = col_data.shape[0]

        # Calculate the index at which the last 20% split starts
        split_index = int(split_rate * total_rows)

        # Condition: The column `y_column` has enough valid values both before and after the split point,
        # and it is not excluded based on other conditions.
        if col_data[y_column].iloc[:split_index].count() >= split_rate * split_index and \
            col_data[y_column].iloc[split_index:].count() >= split_rate * (total_rows - split_index) and \
            building_file not in exclude_file and \
            y_column not in exclude_column:

            model_data = col_data.copy()
            model_data = model_data.rename(columns={y_column: 'y', 'ts': 'ds'})
            model_data = model_data.sort_values(['ds'])

            model_data_path = f'{tmp_path}/{building_file.replace(".csv", "")}_{y_column}_{imputation_method}'

            if save_preprocessed_file == True:
                # Save the original values into a new column
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

    if updated_arguments and model_data_path:
        return updated_arguments
    else:
        return None
