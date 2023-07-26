import csv

def get_add_features(file_path, exclude_column):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        header_row = next(reader)  # Read the first row (header)

        if exclude_column is not None and isinstance(exclude_column, list):
            exclude_columns = exclude_column
            header_row = [column for column in header_row if column not in exclude_columns]

        return header_row
