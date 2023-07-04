import csv
from modules.logging_methods.main import logger

def save_winners_in(results_file_path, winners_in_file_path, config):
    results_header = config['results_header']

    # Load results from the file
    results = []
    with open(results_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        results = list(reader)

    with open(f'{winners_in_file_path}/_winners.in', mode='w') as results_file:
        writer = csv.writer(results_file)

        writer.writerow(results_header)
        
        # Find the indices of the relevant columns
        building_file_index = results_header.index('building_file')
        y_column_index = results_header.index('y_column')
        time_step_index = results_header.index('time_step')
        datelevel_index = results_header.index('datelevel')
        mape_index = results_header.index('mape')
        
        # Group the rows by bldgname and y_column
        rows_by_bldgname_y_column_timestep_datelevel = {}
        for row in results:
            building_file = row[building_file_index].replace('.csv', '')
            y_column = row[y_column_index]
            time_step = row[time_step_index]
            datelevel = row[datelevel_index]
            error = row[mape_index]

            key = (building_file, y_column, time_step, datelevel)
            if key not in rows_by_bldgname_y_column_timestep_datelevel:
                rows_by_bldgname_y_column_timestep_datelevel[key] = []
            rows_by_bldgname_y_column_timestep_datelevel[key].append((error, row))
        
        # Write rows with the lowest error per bldgname and y_column
        for (building_file, y_column, time_step, datelevel), rows in rows_by_bldgname_y_column_timestep_datelevel.items():
            lowest_error = min(rows, key=lambda x: x[0])
            lowest_error_row = lowest_error[1]
            
            # Write the row to the file
            writer.writerow(lowest_error_row)
            
            # Print the lowest error row for each bldgname and y_column
            logger(f"Lowest error for building_file '{building_file}', y_column '{y_column}', time_step, '{time_step}', and datelevel '{datelevel}': {lowest_error[0]}")
            logger('')

            if config['save_at_each_delta']:
                with open(f'{winners_in_file_path}/_winners_{building_file}_{y_column}_{time_step}_{datelevel}.in', mode='w') as results_file:
                    results_writer = csv.writer(results_file)
                    results_writer.writerow(results_header)
                    results_writer.writerow(lowest_error_row)

    return

def save_winners_out(winners_out_file_path, results, config):
    if config['save_at_each_delta']:
        for arg in results:

            building_file = arg['building_file'].replace('.csv', '')
            y_column = arg['y_column']
            time_step = arg['time_step']
            datelevel = arg['datelevel']

            with open(f'{winners_out_file_path}/_winners_{building_file}_{y_column}_{time_step}_{datelevel}.out', mode='w') as results_file:
                results_file.write(str(arg))


    with open(f'{winners_out_file_path}/_winners.out', mode='w') as results_file:
        results_file.write(str(results))

    return


def save_preprocessed_args_results(file_path, results):

    with open(f'{file_path}', mode='w') as results_file:
        results_file.write(str(results))

    return

def save_training_results(file_path, results, winners_in_file_path, config):
    results_header = config['results_header']

    # Convert the results to a set to remove any duplicates
    results = set(results)

    with open(f'{file_path}', mode='w') as results_file:
        writer = csv.writer(results_file)
    
        writer.writerow(results_header)
        writer.writerows(results)

    return
