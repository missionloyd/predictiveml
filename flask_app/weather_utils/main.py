import os, time
from .build_extended_datafile_main import *

def build_extended_clean_data(config):
    csv_files_to_append = []
    csv_files_to_create = []

    path = config['path'] 
    clean_data_path = config['clean_data_path']
  
    for file_name in sorted(os.listdir(clean_data_path), reverse = False):

        file_name_extended = file_name.replace('.csv', '_Extended.csv')

        if file_name.endswith('.csv') and file_name_extended not in config['exclude_file']:
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
        # time.sleep(60 * 2)
    
    # Create the extended datafiles
    for csv_name in csv_files_to_create:
        print(f':: -- Working on {csv_name}.csv ...')
        create_extended_datafile_main(csv_name, config)
    
        data_path = config['data_path']
        clean_data_path = config['clean_data_path']
        file = f'{data_path}/{csv_name}_Extended.csv'
        df_extended = pd.read_csv(file)

        df_extended_modified = df_extended.iloc[:-17]
        df_extended_modified.to_csv(file, mode='w', index=False)

        time.sleep(60 * 2)
    
