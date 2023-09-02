import os

def get_file_names(folder_path, exclude_file):
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            if exclude_file is None or file_name not in exclude_file:
                file_names.append(file_name)
    file_names.sort()
    return file_names
