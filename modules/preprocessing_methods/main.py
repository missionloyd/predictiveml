import pandas as pd
from modules.imputation_methods.main import imputation
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
    preprocess_files = args['preprocess_files']

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

        # Check if column contains the min number of days and is a valid commodity to train on
        if col_data[y_column].count() >= min_number_of_days * 24 and y_column not in exclude_column:

            model_data = col_data.copy()
            model_data = model_data.rename(columns={y_column: 'y', 'ts': 'ds'})
            model_data = model_data.sort_values(['ds'])

            model_data_path = f'{tmp_path}/{building_file.replace(".csv", "")}_{y_column}_{imputation_method}'

            if preprocess_files == True:
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
