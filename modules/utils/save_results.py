import csv

def save_args_results(file_path, results):

    with open(f'{file_path}', mode='w') as results_file:
        results_file.write(str(results))

    return

def save_training_results(file_path, results_header, results, winners_in_file_path):
    # Convert the results to a set to remove any duplicates
    results = set(results)

    with open(f'{file_path}', mode='w') as results_file:
        writer = csv.writer(results_file)
    
        writer.writerow(results_header)
        writer.writerows(results)

    with open(winners_in_file_path, mode='w') as results_file:
        writer = csv.writer(results_file)

        writer.writerow(results_header)
        
        # Find the indices of the relevant columns
        bldgname_index = results_header.index('bldgname')
        y_column_index = results_header.index('y_column')
        mape_index = results_header.index('mape')
        
        # Group the rows by bldgname and y_column
        rows_by_bldgname_y_column = {}
        for row in results:
            bldgname = row[bldgname_index]
            y_column = row[y_column_index]
            error = row[mape_index]
            key = (bldgname, y_column)
            if key not in rows_by_bldgname_y_column:
                rows_by_bldgname_y_column[key] = []
            rows_by_bldgname_y_column[key].append((error, row))
        
        # Write rows with the lowest error per bldgname and y_column
        for (bldgname, y_column), rows in rows_by_bldgname_y_column.items():
            lowest_error = min(rows, key=lambda x: x[0])
            lowest_error_row = lowest_error[1]
            
            # Write the row to the file
            writer.writerow(lowest_error_row)
            
            # Print the lowest error row for each bldgname and y_column
            print(f"Lowest error for bldgname '{bldgname}' and y_column '{y_column}':")
            print(lowest_error_row)
            print()

    return
