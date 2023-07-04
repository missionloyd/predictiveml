#!/usr/bin/env python
# coding: utf-8

import time
import sys
import argparse

arcc_path = '/home/lmacy1/predictiveml'
sys.path.append(arcc_path)  # if running on ARCC
from config import load_config
from modules.utils.create_args import create_args
from modules.utils.process_batch_args import process_batch_args
from modules.preprocessing_methods.main import preprocessing
from weather_utils.main import build_extended_clean_data
from modules.utils.match_args import match_args
from modules.training_methods.main import train_model
from modules.prediction_methods.main import predict
from modules.utils.load_args import load_args
from modules.utils.calculate_duration import calculate_duration
from modules.logging_methods.main import *
from modules.utils.save_args import save_args
from modules.utils.save_results import save_preprocessed_args_results
from modules.utils.save_results import save_training_results

def main(cli_args, job_id_flag, preprocess_flag, train_flag, save_flag, predict_flag):
    start_time = time.time()

    config = load_config(job_id_flag)

    path = config['path']
    batch_size = config['batch_size']
    n_jobs = config['n_jobs']
    update_add_feature = config['update_add_feature']
    args_file_path = f'{path}/models/tmp/_args'
    winners_in_file_path = f'{path}/models/tmp'
    winners_out_file_path = f'{path}/models/tmp'
    results_file_path = f'{path}/results.csv'

    if preprocess_flag or train_flag or save_flag or predict_flag:

        if preprocess_flag:
            # Generate a list of arguments for model training
            args, preprocess_args, config = create_args(config, cli_args)

            # Update files with add_features (weather)
            if update_add_feature:
                build_extended_clean_data(path)

            # Process the preprocessing arguments
            processed_args = process_batch_args('Preprocessing', preprocess_args, preprocessing, batch_size, n_jobs)
            
            # Match processed columns with original argument combinations
            updated_args = match_args(args, processed_args)

            # Save the args to the CSV file
            save_preprocessed_args_results(args_file_path, updated_args)

        if train_flag:
            updated_args = load_args(args_file_path)

            # Process the training arguments
            results = process_batch_args('Training', updated_args, train_model, batch_size, n_jobs)

            # Save the results to the CSV file
            save_training_results(results_file_path, results, winners_in_file_path, config)

        if save_flag: 
            save_args(results_file_path, winners_in_file_path, winners_out_file_path, config)
            winners_file = f'{winners_out_file_path}/_winners.out'
            updated_args = load_args(winners_file)

            # Process the winner training arguments
            results = process_batch_args('Training', updated_args, train_model, batch_size, n_jobs)
                
        if predict_flag:
            required_columns = ['building_file', 'y_column']

            if not all(key in cli_args for key in required_columns):
                logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
                return
            
            # Call the make_prediction function with the provided arguments
            results = predict(cli_args, winners_in_file_path, config)

            # save prediction(s) for info logs and expose to api
            logger(results)
            api_logger(results)
            
    else:
        logger("No flags specified. Exiting...")

    # Calculate duration of the script
    duration = calculate_duration(start_time)

    logger(f"Success... Time elapsed: {duration} hours.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preprocessing, training, or prediction.')
    parser.add_argument('--preprocess', action='store_true', help='Flag for preprocessing.')
    parser.add_argument('--train', action='store_true', help='Flag for training.')
    parser.add_argument('--save', action='store_true', help='Flag for saving tmp files.')
    parser.add_argument('--predict', action='store_true', help='Flag for prediction.')
    parser.add_argument('--building_file', type=str, help='Building file for prediction')
    parser.add_argument('--bldgname', type=str, help='Building name for prediction')
    parser.add_argument('--y_column', type=str, help='Y_Column for prediction')
    parser.add_argument('--startDate', type=str, help='Start date for prediction')
    parser.add_argument('--endDate', type=str, help='End date for prediction')
    parser.add_argument('--time_step', type=int, help='Time step for prediction')
    parser.add_argument('--datelevel', type=str, help='Date level for prediction')
    parser.add_argument('--table', type=str, help='Table for prediction')
    parser.add_argument('--job_id', type=int, help='Flag for logging')

    args = parser.parse_args()
    extract_args()

    preprocess_flag = args.preprocess
    train_flag = args.train
    predict_flag = args.predict
    save_flag = args.save
    job_id_flag = args.job_id

    cli_args = {
        'building_file': args.building_file,
        'bldgname': args.bldgname,
        'y_column': args.y_column,
        'startDate': args.startDate,
        'endDate': args.endDate,
        'datelevel': args.datelevel,
        'time_step': args.time_step,
        'table': args.table,
    }

    if predict_flag:
        main(cli_args, job_id_flag, False, False, False, True)
    else:
        main(cli_args, job_id_flag, preprocess_flag, train_flag, save_flag, False)
