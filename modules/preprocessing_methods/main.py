import pandas as pd
from modules.imputation_methods.main import imputation

def preprocessing(arguments, imputation_methods, building_files_list, header, data_path, min_number_of_days, exclude_column, y_columns):
  
  updated_arguments = list()

  for building_file in building_files_list:
    for method in imputation_methods:
      for y_column in y_columns:
          
          df = pd.read_csv(f'{data_path}/{building_file}')

          # Convert the data into a Pandas dataframe
          df['ts'] = pd.to_datetime(df['ts'])
          df = df.drop_duplicates(subset=['bldgname', 'ts'])
          df = df.sort_values(['bldgname', 'ts'])

          # Group the dataframe by building name and timestamp
          groups = df.groupby('bldgname')
          df = df.set_index('ts')


          # cycle through building names if more than one building per file
          for name, group in groups:
            bldgname = name
            group = group.drop_duplicates(subset=['ts'])

            col_data = group[header]

            # check if column contains the min number of days and is a valid commodity to train on
            if col_data[y_column].count() >= min_number_of_days * 24 and y_column != exclude_column:

              model_data = col_data.copy()
              model_data = model_data.rename(columns={ y_column: 'y', 'ts': 'ds' })
              model_data = model_data.sort_values(['ds'])

              # save the original values into new column
              model_data['y_saved'] = model_data['y']

              # Fill in missing values (preprocessing)
              model_data = imputation(model_data, method)

              for arg in arguments:
                if arg[0] == building_file and arg[1] == method:
                  updated_arg = (model_data, bldgname, *arg)
                  updated_arguments.append(updated_arg)

  return updated_arguments