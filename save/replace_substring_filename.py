import os

# Get the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_list = os.listdir(script_directory)

# Iterate through the files and rename those that match the pattern
for filename in file_list:
    if not filename.endswith(".py"):
        if "_present" in filename:
            new_filename = filename.replace('_present', '_Data_Extended_present')
            file_path = os.path.join(script_directory, filename)
            new_file_path = os.path.join(script_directory, new_filename)
            
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")