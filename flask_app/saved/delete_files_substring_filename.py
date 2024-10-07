import os

# Get the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_list = os.listdir(script_directory)

# Iterate through the files and rename those that match the pattern
for filename in file_list:
    if filename.endswith(".csv"):
        if "year" in filename:
            new_filename = filename.replace('Extended_', 'Extended_spaces_')
            file_path = os.path.join(script_directory, filename)
            
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
