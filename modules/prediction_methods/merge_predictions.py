import os
import pandas as pd
import numpy as np
from modules.db_management.schema_columns_list import schema_columns_list

def merge_predictions(results, config):
    merged_list = []
    y_column_mapping = config['y_column_mapping']

    for df in results:    
        # Extract building_file and remove column
        building_file = df['building_file'].iloc[0]
        df.drop(columns=['building_file'], inplace=True)

        print(f':: -- Generating forecast: {building_file} ...')

        # Open the CSV file and extract the specified columns
        building_file_path = f'{config["data_path"]}/{building_file}'
        csv_data = pd.read_csv(building_file_path)

        # Extracting the desired columns from the CSV data and df
        df.rename(columns={'timestamp': 'ts'}, inplace=True)
        
        df['year'] = df['ts'].dt.year
        df['month'] = df['ts'].dt.month
        df['day'] = df['ts'].dt.day
        df['hour'] = df['ts'].dt.hour

        df['campus'] = csv_data['campus'].iloc[0]
        df['latitude'] = csv_data['latitude'].iloc[0]
        df['longitude'] = csv_data['longitude'].iloc[0]

        for key, _ in y_column_mapping.items():
            historical_key = key.replace('present', 'historical')

            # Filter csv_data to include only timestamps present in df
            filtered_csv_data = csv_data[csv_data['ts'].isin(df['ts'].dt.strftime(config['datetime_format']))]

            # Convert 'ts' column to datetime in filtered_csv_data
            filtered_csv_data = filtered_csv_data.copy()  
            filtered_csv_data['ts'] = pd.to_datetime(filtered_csv_data['ts'])

            # Perform the merge using the common 'ts' column
            merged_data = pd.merge(
                df[['ts']],
                filtered_csv_data[['ts', key, historical_key]],
                on='ts',
                how='left'
            )
            # Replace NaN with None
            merged_data = merged_data.applymap(lambda x: None if pd.isna(x) else x)

            # Update the values in df with the values from csv_data
            df[key] = merged_data[key]
            df[historical_key] = merged_data[historical_key]

            # df[key] = None
            # df[historical_key] = None

        df['ts'] = df['ts'].dt.strftime(config['datetime_format'])
        df.replace(to_replace=[np.nan, "None"], value=None, inplace=True)        
        df = df.reindex(columns=schema_columns_list())

        merged_list.append({ building_file: df })
        # print(df)
  
    return merged_list