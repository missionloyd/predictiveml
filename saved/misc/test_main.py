#!/usr/bin/env python
# coding: utf-8

import time
import sys
from concurrent.futures import ThreadPoolExecutor

arcc_path = '/home/lmacy1/predictiveml'
sys.path.append(arcc_path)  # if running on ARCC
from config import load_config
from modules.utils.create_args import create_args
from modules.utils.process_batch_args import process_batch_args
from modules.preprocessing_methods.main import preprocessing
from modules.utils.match_args import match_args
from modules.training_methods.main import train_model
from modules.utils.save_results import save_results
from modules.utils.calculate_duration import calculate_duration

if __name__ == '__main__':
    start_time = time.time()

    config = load_config()

    path = config['path']
    results_header = config['results_header']
    batch_size = config['batch_size']
    n_jobs = config['n_jobs']

    # # Generate a list of arguments for model training
    # args, preprocess_args = create_args(config)

    # # Process the preprocessing arguments
    # with ThreadPoolExecutor() as executor:
    #     processed_args = executor.submit(process_batch_args, 'Preprocessing', preprocess_args, preprocessing, batch_size, n_jobs).result()

    # # Match processed columns with original argument combinations
    # updated_args = match_args(args, processed_args)

    updated_args = [{'model_data_path': './models/tmp/Stadium_Data_Extended_present_elec_kwh_linear_interpolation', 'bldgname': 'Stadium', 'building_file': 'Stadium_Data_Extended.csv', 'y_column': 'present_elec_kwh', 'imputation_method': 'linear_interpolation', 'model_type': 'xgboost', 'feature_method': 'lassocv', 'n_feature': 1, 'time_step': 24, 'header': ['ts', 'present_elec_kwh', 'temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg'], 'data_path': './clean_data_extended', 'add_feature': ['temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg'], 'min_number_of_days': 365, 'exclude_column': ['present_co2_tons'], 'n_fold': 5, 'split_rate': 0.8, 'minutes_per_model': 2, 'memory_limit': 102400, 'save_model_file': False, 'save_model_plot': True, 'path': '.', 'preprocess_files': False}]

    # Process the training arguments
    with ThreadPoolExecutor() as executor:
        results = executor.submit(process_batch_args, 'Training', updated_args, train_model, batch_size, n_jobs).result()

    # Convert the results to a set to remove any duplicates
    unique_results = set(results)

    # Save the results to the CSV file
    save_results(path, results_header, unique_results)

    # Calculate duration of the script
    duration = calculate_duration(start_time)

    print(f"Success... Time elapsed: {duration} hours.")
