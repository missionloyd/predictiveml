import os

def save_predictions(merged_list, config):
    path = config['path']
    file_path = f'{path}/prediction_data'

    for item in merged_list:
        for building_file_name, df in item.items():
            building_file_path = os.path.join(file_path, building_file_name)

            if os.path.exists(building_file_path):
                df.to_csv(building_file_path, mode='a', index=False, header=False)
            else:
                df.to_csv(building_file_path, mode='w', index=False, header=True)

    return