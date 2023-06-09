import os

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    file_names.sort()
    return file_names

# Example usage
# folder_path = "../../clean_data_extended"
# file_names = get_file_names(folder_path)
# for file_name in file_names:
#     print(file_name)
