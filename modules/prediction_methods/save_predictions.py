import os
import pandas as pd

def save_predictions(merged_list, config, cli_args):
    path = config['path']
    results_file = config['results_file']
    datelevel = 'hour'
    time_step = 24

    if len(config['datelevel']) > 0:
        datelevel = str(config['datelevel'][0])

    if len(config['time_step']) > 0:
        time_step = str(config['time_step'][0])

    datelevel = cli_args['datelevel'] or datelevel
    time_step = str(cli_args['time_step']) or time_step
    file_path = f'{path}/prediction_data'

    building_data_dict = {}

    for item in merged_list:
        for building_file_name, df in item.items():
            all_y_columns =  config['y_column'] + ['present_co2_tonh']
            historical_columns = [col.replace('present', 'historical') for col in all_y_columns]
            drop_columns = all_y_columns + historical_columns
            df.drop(columns=drop_columns, inplace=True)
            building_data_dict[building_file_name] = df

    for building_file_name, df in building_data_dict.items():
        # print(building_file_name)
        modified_building_file_name = building_file_name.replace('.csv', '')
        modified_building_file_name = f'{modified_building_file_name}_{results_file}'
        building_file_path = os.path.join(file_path, modified_building_file_name)

        # print(df)

        if os.path.exists(building_file_path):
            existing_df = pd.read_csv(building_file_path)
            merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=['ts'])
            merged_df.to_csv(building_file_path, index=False)
        else:
            df.to_csv(building_file_path, index=False)

    return