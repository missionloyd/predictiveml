import os

def save_predictions(merged_list, config, cli_args):
    path = config['path']
    datelevel = 'hour'
    time_step = 48

    if len(config['datelevel']) != '':
        datelevel = str(config['datelevel'][0])

    if len(config['time_step']) > 0:
        time_step = str(config['time_step'][0])

    datelevel = cli_args['datelevel'] or datelevel
    time_step = str(cli_args['time_step']) or time_step
    file_path = f'{path}/prediction_data'

    for item in merged_list:
        for building_file_name, df in item.items():
            modified_building_file_name = building_file_name.replace('.csv', '')
            modified_building_file_name = f'{modified_building_file_name}_{time_step}_{datelevel}.csv'
            building_file_path = os.path.join(file_path, modified_building_file_name)

            if os.path.exists(building_file_path):
                df.to_csv(building_file_path, mode='a', index=False, header=False)
            else:
                df.to_csv(building_file_path, mode='w', index=False, header=True)

    return