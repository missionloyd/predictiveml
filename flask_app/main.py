#!/usr/bin/env python
# coding: utf-8

import time
import argparse
import sys

arcc_path = '/home/lmacy1/predictiveml'
sys.path.append(arcc_path)  # if running on ARCC

from config import load_config
from modules.utils.update_config import update_config
from modules.logging_methods.main import setup_logger
from modules.utils.prune import prune
from modules.utils.calculate_duration import calculate_duration
from modules.logging_methods.main import logger, extract_args

from modules.main_functions.update_add_feature import update_add_feature
from modules.main_functions.preprocess import preprocess
from modules.main_functions.train import train
from modules.main_functions.save import save
from modules.main_functions.predict import predict
from modules.main_functions.saved_predict import saved_predict
from modules.main_functions.save_predictions import save_predictions
from modules.main_functions.insert import insert

def main(cli_args, flags):
    start_time = time.time()

    run_all_flag = flags['run_all']
    prune_flag = flags['prune_flag']
    update_add_feature_flag = flags['update_add_feature_flag']
    predict_flag = flags['predict_flag']
    saved_predict_flag = flags['saved_predict_flag']
    preprocess_flag = flags['preprocess_flag']
    train_flag = flags['train_flag'] 
    save_flag = flags['save_flag']
    insert_flag = flags['insert_flag']
    save_predictions_flag = flags['save_predictions_flag']

    if run_all_flag or prune_flag or update_add_feature_flag or preprocess_flag or train_flag or save_flag or predict_flag or saved_predict_flag or insert_flag or save_predictions_flag or insert_flag:

        path = cli_args['path']
        data_path = cli_args['data_path']
        clean_data_path = cli_args['clean_data_path']
        table = cli_args['table']

        # Update config if user provided any arguments
        local_config = load_config(path, data_path, clean_data_path, table)
        config = update_config(local_config, cli_args)

        # Setup logger and clean files/folders specified in config
        setup_logger(config)
        prune(prune_flag, config)

        # Update data files with additional columns (weather)
        if update_add_feature_flag:
            update_add_feature(config)

        # Fill in missing gaps in the data and generate a list of arguments to enable training
        if preprocess_flag or run_all_flag:
            preprocess(config)

        # Train on preprocessed data
        if train_flag or run_all_flag:
            train(config)

        # Save model files to enable predictions
        if save_flag or run_all_flag: 
            save(config)

        # Predict using weather columns as features or make a forecast without using weather data 
        # (leave --startDateTime and endDateTime as '' to make a forecast)
        if predict_flag:
            predict(config, cli_args)

        # Load saved predictions and jsonify (custom)
        if saved_predict_flag:
            saved_predict(config, cli_args)

        # Save predictions to new folder
        if save_predictions_flag:
            save_predictions(config, cli_args)
            
        # Predict on all available models and then insert into database (custom)
        if insert_flag:
            insert(config, cli_args)

    else:
        logger("No flags specified. Exiting...")

    # Calculate duration of the script
    hours, minutes, seconds = calculate_duration(start_time)

    logger(f"\nSuccess... Time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preprocessing, training, or predicting on timeseries data.')
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
    parser.add_argument('--model_type', type=str, help='Model type for training and predicting.')
    parser.add_argument('--imputation_method', type=str, help='Imputation method for preprocessing.')
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
        'model_type': args.model_type,
        'imputation_method': args.imputation_method,
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