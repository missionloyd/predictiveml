import os

def get_file_names(folder_path, exclude_file):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            if exclude_file is None or file_name not in exclude_file:
                file_names.append(file_name)
    file_names.sort()
    return file_names
