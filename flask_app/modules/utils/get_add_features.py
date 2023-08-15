import csv, os, sys

def get_add_features(file_list, data_path, exclude_column):
    file_path = ''
    
    if len(file_list) > 0:
        file_path = f'{data_path}/{file_list[0]}'
    
    if os.path.exists(file_path) and len(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            header_row = next(reader)  # Read the first row (header)

            if exclude_column is not None and isinstance(exclude_column, list):
                exclude_columns = exclude_column
                header_row = [column for column in header_row if column not in exclude_columns]

            return header_row
    else:
        return list()