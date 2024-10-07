import os, json

# Get the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_list = os.listdir(script_directory)

building_names = []

# Iterate through the files and rename those that match the pattern
for filename in file_list:
    if filename.endswith(".csv"):
        building_names.append(filename.replace('_Extended_spaces_hour.csv', '').replace('_Extended_spaces_day.csv', '').replace('_Extended_spaces_month.csv', '').replace('_Extended_spaces_year.csv', ''))

building_names = list(set(building_names))

with open('buildingListItems.json', 'r') as f:
    tree_nodes = json.load(f)
    new_field_name = 'forecast_available'

    for node in tree_nodes:
        b_name = node['link'].replace('/', '')

        if b_name in building_names:
            node[new_field_name] = True
        else:
            node[new_field_name] = False

with open('test', 'w') as f:
    f.write(json.dumps(tree_nodes, indent=2))
