import os, json
import pandas as pd

# Get the current directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_list = os.listdir(script_directory)

buildings = []

# Iterate through the files and rename those that match the pattern
for filename in file_list:
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        buildings.append({ 'name': filename, 'campus': df['campus'][0] })

with open('campusListItems.json', 'r') as c:
    head_tree_nodes = json.load(c)

    with open('buildingListItems.json', 'r') as b:
        tree_nodes = json.load(b)

        for head_node in head_tree_nodes:

            building_list = list()

            for building in tree_nodes:
                print(building)
                if head_node['campus'] == building['campus']:

                    building_list.append({
                        "name": building['name'],
                        "longName": building['longName'],
                        "link": building['link'],
                        "query": building['query'],
                        "spaceid": building['spaceid'],
                        "slug": building['slug'],
                        "Electricity_plotid": building['Electricity_plotid'],
                        "Hot_Water_plotid": building['Hot_Water_plotid'],
                        "Water_plotid": building['Water_plotid'],
                        "Chilled_Water_plotid": building['Chilled_Water_plotid'],
                        "imageid": building['imageid'],
                        "forecast_available": building['forecast_available'],
                        "campus": building['campus'],
                        "children": []
                    })

            head_node['children'] = building_list

        # for node in tree_nodes:
        #     if not node[new_field_name]:
        #         print(node['bldgname'])


print(json.dumps(head_tree_nodes, indent=2)) 

with open('listItems.json', 'w') as f:
    f.write(json.dumps(head_tree_nodes, indent=2))
