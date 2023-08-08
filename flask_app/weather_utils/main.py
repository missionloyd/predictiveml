import os
from .build_extended_datafile_main import *

#csv_name = 'Science_Initiative_Building_Data'
#build_extended_datafile(csv_name)

# Check folder 'clean_data' for *.csv files and create a list of the filenames

def build_extended_clean_data(path):
  csv_files_to_work = []
  for file_name in sorted(os.listdir(f'{path}/clean_data'), reverse = False):
      if file_name.endswith('.csv'):
          # Check if file is already present inside the 'clean_data_extended' folder
          if not os.path.isfile(f'../clean_data_extended/{file_name[:-4]}_Extended.csv'):
              csv_files_to_work.append(file_name[:-4]) # Remove the '.csv' extension

  # print(csv_files_to_work)
  # print(len(csv_files_to_work))

  # Build the extended datafiles
  for csv_name in csv_files_to_work:
      print(f':: -- Working on {csv_name}.csv ...')
      append_to_extended_datafile_main(csv_name, path)
