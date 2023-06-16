import csv

def save_results(path, results_header, results):
    with open(f'{path}/results.csv', mode='w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(results_header)
        writer.writerows(results)

    # Find the index of the 'mae' column
    mae_index = results_header.index('mae')
    
    # Find the row with the smallest 'mae' value
    lowest_error = min(results, key=lambda row: row[mae_index])

    print('Model with the lowest error (MAE):')
    print(lowest_error)
    print()

    return