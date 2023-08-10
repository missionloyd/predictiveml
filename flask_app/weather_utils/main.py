import os
from .build_extended_datafile_main import *

def build_extended_clean_data(config):
    csv_files_to_append = []
    csv_files_to_create = []

    path = config['path'] 
    clean_data_path = config['clean_data_path']
  
    for file_name in sorted(os.listdir(clean_data_path), reverse = False):
        if file_name.endswith('.csv'):
            # Check if file is already present inside the 'clean_data_extended' folder
            data_path = config['data_path']

            if os.path.isfile(f'{data_path}/{file_name[:-4]}_Extended.csv'):
                csv_files_to_append.append(file_name[:-4]) # Remove the '.csv' extension
            else:
                csv_files_to_create.append(file_name[:-4])

    # Append the extended datafiles
    for csv_name in csv_files_to_append:
        print(f':: -- Working on {csv_name}.csv ...')
        append_to_extended_datafile_main(csv_name, config)
        break
    
    # # Create the extended datafiles
    # for csv_name in csv_files_to_create:
    #     print(f':: -- Working on {csv_name}.csv ...')
    #     create_extended_datafile_main(csv_name, config)
    
