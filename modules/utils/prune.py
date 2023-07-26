import shutil, os, csv

def prune(prune_flag, directories_to_prune, config):
    for directory in directories_to_prune:
        if os.path.exists(directory):
            if prune_flag:
                shutil.rmtree(directory)
                os.makedirs(directory)
                with open(config['results_file_path'], 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(config['results_header'])
                    
        else:
            os.makedirs(directory)
            if prune_flag:
                with open(config['results_file_path'], 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(config['results_header'])
    return