#!/usr/bin/env python
# coding: utf-8

import time
import sys
import argparse

arcc_path = '/home/lmacy1/predictiveml'
sys.path.append(arcc_path)  # if running on ARCC
from config import load_config
from modules.utils.update_config import update_config
from modules.logging_methods.main import setup_logger
from modules.utils.create_args import create_args
from modules.utils.process_batch_args import process_batch_args
from modules.utils.prune import prune
from modules.preprocessing_methods.main import preprocessing
from modules.utils.adaptive_sampling import adaptive_sampling
from weather_utils.main import build_extended_clean_data
from modules.utils.match_args import match_args
from modules.training_methods.main import train_model
from modules.prediction_methods.on_demand_prediction import on_demand_prediction
from modules.prediction_methods.master_prediction import master_prediction
from modules.utils.load_args import load_args
from modules.utils.calculate_duration import calculate_duration
from modules.logging_methods.main import logger, api_logger, extract_args
from modules.utils.save_args import save_args
from modules.utils.save_results import save_preprocessed_args_results
from modules.utils.save_results import save_training_results
from modules.db_management.insert_data import insert_data

def main(cli_args, flags):
    start_time = time.time()

    run_all_flag = flags['run_all']
    prune_flag = flags['prune_flag']
    predict_flag = flags['predict_flag']
    preprocess_flag = False if flags['predict_flag'] else flags['preprocess_flag']
    train_flag = False if flags['predict_flag'] else flags['train_flag'] 
    save_flag = False if flags['predict_flag'] else flags['save_flag']
    insert_flag = False if flags['predict_flag'] else flags['insert_flag']

    if run_all_flag or prune_flag or preprocess_flag or train_flag or save_flag or predict_flag or insert_flag or insert_flag:

        config = load_config()
        config = update_config(config, cli_args)
        setup_logger(config)
        path = config['path']
        batch_size = config['batch_size']
        n_jobs = config['n_jobs']
        results_file_path = config['results_file_path']
        update_add_feature = config['update_add_feature']
        args_file_path = f'{path}/models/tmp/_args'
        winners_in_file_path = f'{path}/models/tmp'
        winners_out_file_path = f'{path}/models/tmp'
        winners_out_file = f'{winners_out_file_path}/_winners.out'

        prune(prune_flag, config)

        if preprocess_flag or run_all_flag:
            # Generate a list of arguments for model training
            args, preprocess_args = create_args(config)
            args = adaptive_sampling(args, config)

            # Update files with add_features (weather)
            if update_add_feature:
                build_extended_clean_data(path)

            # Process the preprocessing arguments
            processed_args = process_batch_args('Preprocessing', preprocess_args, preprocessing, batch_size, n_jobs, config)
            
            # Match processed columns with original argument combinations
            updated_args = match_args(args, processed_args)

            # Save the args to the CSV file
            save_preprocessed_args_results(args_file_path, updated_args)

        if train_flag or run_all_flag:
            updated_args = load_args(args_file_path)

            # Process the training arguments
            results = process_batch_args('Training', updated_args, train_model, batch_size, n_jobs, config)

            # Save the results to the CSV file
            save_training_results(results_file_path, results, config)

        if save_flag or run_all_flag: 
            save_args(results_file_path, winners_in_file_path, winners_out_file_path, config)
            updated_args = load_args(winners_out_file)

            # Re-train winners and save the model file
            results = process_batch_args('Saving', updated_args, train_model, batch_size, n_jobs, config)
                
        if predict_flag:
            required_columns = ['building_file', 'y_column']

            if not all(key in cli_args for key in required_columns):
                logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
                return
            
            # Call the make_prediction function with the provided arguments
            results = on_demand_prediction(cli_args, winners_in_file_path, config)

            # save prediction(s) for info logs and expose to api
            logger(results)
            api_logger(results)

        if insert_flag:
            # Predict on all available models and then insert into database (custom for campus heartbeat)
            results = master_prediction(cli_args, winners_in_file_path, config)
            insert_data(results, config)
            
    else:
        logger("No flags specified. Exiting...")

    # Calculate duration of the script
    duration = calculate_duration(start_time)

    logger(f"Success... Time elapsed: {duration} hours.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preprocessing, training, or predicting.')
    parser.add_argument('--run_all', action='store_true', help='Flag for overriding all other flags except --prune, --predict, and --insert.')
    parser.add_argument('--prune', action='store_true', help='Flag for pruning specific directories.')  
    parser.add_argument('--preprocess', action='store_true', help='Flag for preprocessing.')
    parser.add_argument('--train', action='store_true', help='Flag for training.')
    parser.add_argument('--save', action='store_true', help='Flag for saving model files.')
    parser.add_argument('--predict', action='store_true', help='Flag for prediction.')
    parser.add_argument('--insert', action='store_true', help='Flag for inserting predictings into database.') 
    parser.add_argument('--building_file', type=str, help='Building file for prediction (do not include .csv extension).')
    parser.add_argument('--bldgname', type=str, help='Building name for prediction.')
    parser.add_argument('--y_column', type=str, help='Y_Column for prediction.')
    parser.add_argument('--startDateTime', type=str, help='Start date for prediction.')
    parser.add_argument('--endDateTime', type=str, help='End date for prediction.')
    parser.add_argument('--time_step', type=int, help='Time step for prediction.')
    parser.add_argument('--datelevel', type=str, help='Date level for prediction.')
    parser.add_argument('--table', type=str, help='Table for prediction.')
    parser.add_argument('--save_preprocessed_files', action='store_true', help='Flag for saving/reusing preprocessed files.')
    parser.add_argument('--temperature', type=float, help='Temperature for training.') 
    parser.add_argument('--job_id', type=int, help='Job ID for logging.')

    args = parser.parse_args()
    extract_args()

    flags = {
        'run_all': args.run_all,
        'prune_flag': args.prune,
        'insert_flag': args.insert,
        'preprocess_flag': args.preprocess,
        'train_flag': args.train,
        'predict_flag': args.predict,
        'save_flag': args.save,
    }

    cli_args = {
        'job_id': args.job_id,
        'building_file': args.building_file,
        'bldgname': args.bldgname,
        'y_column': args.y_column,
        'startDateTime': args.startDateTime,
        'endDateTime': args.endDateTime,
        'datelevel': args.datelevel,
        'time_step': args.time_step,
        'table': args.table,
        'save_preprocessed_files': args.save_preprocessed_files,
    }

    main(cli_args, flags)