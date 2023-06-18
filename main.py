#!/usr/bin/env python
# coding: utf-8

import time
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

arcc_path = '/home/lmacy1/predictiveml'
sys.path.append(arcc_path)  # if running on ARCC
from config import load_config
from modules.utils.create_args import create_args
from modules.utils.process_batch_args import process_batch_args
from modules.preprocessing_methods.main import preprocessing
from modules.utils.match_args import match_args
from modules.training_methods.main import train_model
from modules.utils.load_args import load_args
from modules.utils.save_results import save_results
from modules.utils.calculate_duration import calculate_duration

def main(preprocess_flag):
    start_time = time.time()

    config = load_config()

    path = config['path']
    results_header = config['results_header']
    batch_size = config['batch_size']
    n_jobs = config['n_jobs']
    args_file_path = f'{path}/models/tmp/_results.txt'
    results_file_path = f'{path}/results.csv'

    if preprocess_flag:
        # Generate a list of arguments for model training
        args, preprocess_args = create_args(config)

        # Process the preprocessing arguments
        processed_args = process_batch_args('Preprocessing', preprocess_args, preprocessing, batch_size, n_jobs)
        
        # Match processed columns with original argument combinations
        updated_args= match_args(args, processed_args)

        # Save the args to the CSV file
        save_results(args_file_path, results_header, updated_args, preprocess_flag)
    else:
        updated_args = load_args(args_file_path)

        # Process the training arguments
        results = process_batch_args('Training', updated_args, train_model, batch_size, n_jobs)

        # Save the results to the CSV file
        save_results(results_file_path, results_header, results, preprocess_flag)

    # Calculate duration of the script
    duration = calculate_duration(start_time)

    print(f"Success... Time elapsed: {duration} hours.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preprocessing or training.')
    parser.add_argument('--preprocess', action='store_true', help='Flag for preprocessing.')

    args = parser.parse_args()

    preprocess_flag = args.preprocess

    main(preprocess_flag)
