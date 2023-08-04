import ast

def load_args(file_path):
    updated_args = []

    with open(f'{file_path}', 'r') as file:
        content = file.read()
        updated_args = ast.literal_eval(content)

    return updated_args