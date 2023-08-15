#!/usr/bin/env python
# coding: utf-8

import time
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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

def process_preprocessing_args(preprocess_args, preprocessing, batch_size, n_jobs):
    return process_batch_args(
        'Preprocessing',
        preprocess_args,
        preprocessing,
        batch_size,
        n_jobs
    )

def process_training_args(updated_args, train_model, batch_size, n_jobs):
    return process_batch_args(
        'Training',
        updated_args,
        train_model,
        batch_size,
        n_jobs
    )

if __name__ == '__main__':
    start_time = time.time()

    config = load_config()

    path = config['path']
    results_header = config['results_header']
    batch_size = config['batch_size']
    n_jobs = config['n_jobs']

    # Generate a list of arguments for model training
    args, preprocess_args = create_args(config)

    with ProcessPoolExecutor() as executor:
        # Process the preprocessing arguments
        preprocess_fn = partial(
            process_preprocessing_args,
            preprocess_args,
            preprocessing,
            batch_size,
            n_jobs
        )
        processed_args = executor.submit(preprocess_fn).result()

        # Match processed columns with original argument combinations
        updated_args = match_args(args, processed_args)

        # Process the training arguments
        training_fn = partial(
            process_training_args,
            updated_args,
            train_model,
            batch_size,
            n_jobs
        )
        results = executor.submit(training_fn).result()

    # Convert the results to a set to remove any duplicates
    unique_results = set(results)

    # Save the results to the CSV file
    save_results(path, results_header, unique_results)

    # Calculate duration of the script
    duration = calculate_duration(start_time)

    print(f"Success... Time elapsed: {duration} hours.")
