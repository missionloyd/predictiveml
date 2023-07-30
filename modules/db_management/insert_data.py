import os
import pandas as pd
from .db_management import populate_table
from .schema_columns_list import schema_columns_list

def insert_data(results, config):
    table = config['table']
    y_column_mapping = config['y_column_mapping']

    for df in results:
        # Extract building_file and remove column
        building_file = df['building_file'].iloc[0]
        df.drop(columns=['building_file'], inplace=True)

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

            df[key] = None
            df[historical_key] = None

        df = df.reindex(columns=schema_columns_list())

        print(df)

    # populate_table(table_name=table, df=spaces_df)
  
    return
