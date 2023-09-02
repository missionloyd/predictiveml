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
from modules.prediction_methods.saved_on_demand_prediction import saved_on_demand_prediction
from modules.utils.load_args import load_args
from modules.utils.calculate_duration import calculate_duration
from modules.logging_methods.main import logger, api_logger, extract_args
from modules.utils.save_args import save_args
from modules.utils.save_results import save_preprocessed_args_results
from modules.utils.save_results import save_training_results
from modules.prediction_methods.merge_predictions import merge_predictions
from modules.prediction_methods.save_predictions import save_predictions
from modules.db_management.insert_data import insert_data

def main(cli_args, flags):
    start_time = time.time()

    run_all_flag = flags['run_all']
    prune_flag = flags['prune_flag']
    update_add_feature_flag = flags['update_add_feature_flag']
    predict_flag = flags['predict_flag']
    saved_predict_flag = flags['saved_predict_flag']
    preprocess_flag = False if flags['predict_flag'] else flags['preprocess_flag']
    train_flag = False if flags['predict_flag'] else flags['train_flag'] 
    save_flag = False if flags['predict_flag'] else flags['save_flag']
    insert_flag = False if flags['predict_flag'] else flags['insert_flag']
    save_predictions_flag = False if flags['predict_flag'] else flags['save_predictions_flag']

    if run_all_flag or prune_flag or update_add_feature_flag or preprocess_flag or train_flag or save_flag or predict_flag or saved_predict_flag or insert_flag or save_predictions_flag or insert_flag:

        path = cli_args['path']
        data_path = cli_args['data_path']
        clean_data_path = cli_args['clean_data_path']
        table = cli_args['table']
        config = load_config(path, data_path, clean_data_path, table)
        config = update_config(config, cli_args)
        setup_logger(config)
        path = config['path']
        batch_size = config['batch_size']
        n_jobs = config['n_jobs']
        results_file_path = config['results_file_path']
        args_file_path = config['args_file_path']
        winners_in_file_path = config['winners_in_file_path']
        winners_out_file_path = config ['winners_out_file_path']
        winners_out_file = config['winners_out_file']

        prune(prune_flag, config)

        # Update files with add_features (weather)
        if update_add_feature_flag:
            build_extended_clean_data(config)

        if preprocess_flag or run_all_flag:
            # Generate a list of arguments for model training
            args, preprocess_args = create_args(config)
            args = adaptive_sampling(args, config)

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
            required_columns = ['building_file', 'y_column', 'time_step', 'datelevel']

            if not all(key in cli_args for key in required_columns):
                logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
                return
            
            # Call the make_prediction function with the provided arguments
            results = on_demand_prediction(cli_args, winners_in_file_path, config)

            # save prediction(s) for info logs and expose to api
            logger(results)
            api_logger(results, config)

        if saved_predict_flag:
            required_columns = ['time_step', 'datelevel']

            if not all(key in cli_args for key in required_columns):
                logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
                return
            
            results = saved_on_demand_prediction(cli_args, config)
            logger(results)
            api_logger(results, config)

        # Save predictions merged with original datasets
        if save_predictions_flag:
            results = master_prediction(cli_args, winners_in_file_path, config)
            merged_list = merge_predictions(results, config)
            save_predictions(merged_list, config, cli_args)
            
        # Predict on all available models and then insert into database (custom for campus heartbeat)
        if insert_flag:
            results = master_prediction(cli_args, winners_in_file_path, config)
            merged_list = merge_predictions(results, config)
            insert_data(merged_list, config)
   
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
    parser.add_argument('--saved_predict', action='store_true', help='Flag for loading saved local prediction.')
    parser.add_argument('--insert', action='store_true', help='Flag for inserting predictings into database.') 
    parser.add_argument('--save_predictions', action='store_true', help='Flag for saving predictions.') 
    parser.add_argument('--update_add_feature', action='store_true', help='Flag for updating additional features into database.') 
    parser.add_argument('--building_file', type=str, help='Building file for prediction (do not include .csv extension).')
    parser.add_argument('--bldgname', type=str, help='Building name for prediction.')
    parser.add_argument('--y_column', type=str, help='Y_Column for prediction.')
    parser.add_argument('--startDateTime', type=str, help='Start date for prediction.')
    parser.add_argument('--endDateTime', type=str, help='End date for prediction.')
    parser.add_argument('--time_step', type=int, help='Time step for prediction.')
    parser.add_argument('--datelevel', type=str, help='Date level for prediction.')
    parser.add_argument('--table', type=str, help='Table for prediction.')
    parser.add_argument('--results_file', type=str, help='File name for training results.')
    parser.add_argument('--save_preprocessed_files', action='store_true', help='Flag for saving/reusing preprocessed files.')
    parser.add_argument('--temperature', type=float, help='Temperature for training.') 
    parser.add_argument('--job_id', type=int, help='Job ID for logging.')
    parser.add_argument('--path', type=str, help='Path of root project directory.')
    parser.add_argument('--data_path', type=str, help='Path to training data folder.')
    parser.add_argument('--clean_data_path', type=str, help='Path to preprocessing data folder.')

    args = parser.parse_args()
    extract_args()

    flags = {
        'run_all': args.run_all,
        'prune_flag': args.prune,
        'insert_flag': args.insert,
        'preprocess_flag': args.preprocess,
        'train_flag': args.train,
        'predict_flag': args.predict,
        'saved_predict_flag': args.saved_predict,
        'save_flag': args.save,
        'save_predictions_flag': args.save_predictions,
        'update_add_feature_flag': args.update_add_feature,
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
        'results_file': args.results_file,
        'temperature': args.temperature,
        'save_preprocessed_files': args.save_preprocessed_files,
        'path': args.path,
        'data_path': args.data_path,
        'clean_data_path': args.clean_data_path,
    }

    main(cli_args, flags)