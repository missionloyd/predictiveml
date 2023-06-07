import csv

def save_results(path, results_header, results):
    with open(f'{path}/results.csv', mode='w') as results_file:
      writer = csv.writer(results_file)
      writer.writerow(results_header)
      writer.writerows(results)

    return