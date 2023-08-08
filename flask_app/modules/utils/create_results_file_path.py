import os

def create_results_file_path(path, results_file):
    results_directory ='results'
    os.makedirs(results_directory, exist_ok=True)

    return f'{path}/{results_directory}/{results_file}'