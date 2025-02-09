import shutil, os, csv

def prune(prune_flag, config):

    temperature = config['temperature']

    for directory in config['directories_to_prune']:
        if os.path.exists(directory):
            if prune_flag:
                shutil.rmtree(directory)
                os.makedirs(directory)             
        else:
            os.makedirs(directory)

    if (prune_flag and not (temperature > 0 and temperature <= 1.0)) or not os.path.exists(config['results_file_path']):
        with open(config['results_file_path'], 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(config['results_header'])
    
    return