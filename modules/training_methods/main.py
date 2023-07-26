import logging_config
import csv
import pickle
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from autosklearn.regression import AutoSklearnRegressor
import xgboost as xgb

from modules.utils.resample_data import resample_data
from modules.utils.detect_data_frequency import detect_data_frequency
from modules.feature_methods.main import feature_engineering
from modules.logging_methods.main import logger

# Define the function to process each combination of parameters
def train_model(args, config):
    model_data_path = args['model_data_path']
    bldgname = args['bldgname']
    building_file = args['building_file']
    y_column = args['y_column']
    imputation_method = args['imputation_method']
    model_type = args['model_type']
    feature_method = args['feature_method']
    n_feature = args['n_feature']
    updated_n_feature = args['updated_n_feature']
    time_step = args['time_step']
    datelevel = args['datelevel']
    header = config['header']
    data_path = config['data_path']
    add_feature = config['add_feature']
    selected_features_delimited = args['selected_features_delimited']
    exclude_column = config['exclude_column']
    n_fold = config['n_fold']
    train_test_split = config['train_test_split']
    minutes_per_model = config['minutes_per_model']
    memory_limit = config['memory_limit']
    save_model_file = args['save_model_file']
    save_model_plot = config['save_model_plot']
    path = config['path']

    # Check if the file exists
    if not os.path.exists(model_data_path):
        logger("File not found. Please set save_preprocessed_file: True and run with --preprocess")
        sys.exit()

    # Load model_data separately within each task
    with open(model_data_path, 'rb') as file:
        model_data = pickle.load(file)

    out_path = f'{path}/models/{model_type}'

    original_datelevel = detect_data_frequency(model_data)

    # This code aggregates and resamples a DataFrame based on given datelevel
    model_data = resample_data(model_data, datelevel, original_datelevel)

    # normalize the data, save orginal data column for graphing later
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))

    # normalize additional features
    add_data_scaled = np.empty((model_data.shape[0], 0))

    for feature in add_feature:
        feature_scaler = StandardScaler()
        add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
        add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

    # check if selected_features are already saved
    if selected_features_delimited:
        selected_features = selected_features_delimited.split('|')
    else:
        # identify most important features and eliminate less important features
        selected_features = feature_engineering(feature_method, n_fold, add_data_scaled, data_scaled, add_feature)
        selected_features_delimited = '|'.join(selected_features)
        updated_n_feature = n_feature

    # normalize selected features
    add_data_scaled = np.empty((model_data.shape[0], 0))

    if len(selected_features) > 0:
        for feature in selected_features:
            feature_scaler = StandardScaler()
            add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
            add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

        # ensures that updated_n_feature does not exceed the number of selected features or the number of samples in add_data_scaled
        max_nfeatures_nsamples = min(add_data_scaled.shape[0], add_data_scaled.shape[1])
        if updated_n_feature > max_nfeatures_nsamples:
            updated_n_feature = max_nfeatures_nsamples

        # train PCA (Linear Dimensionality Reduction) with multi-feature output
        pca = PCA(n_components=updated_n_feature)
        pca_data = pca.fit_transform(add_data_scaled)
        data_scaled = np.concatenate((data_scaled, pca_data), axis=1)
    else:
        # Handle the case where no features are selected
        updated_n_feature = 0
    
    # split the data into training and testing sets
    train_size = int(len(data_scaled) * train_test_split)
    train_data = data_scaled[:train_size, :]
    test_data = data_scaled[train_size:, :]

    # save y values for benchmarking/plotting
    y_test = model_data['y'].iloc[train_size:].reset_index(drop=True)
    saved_y_test = model_data['y_saved'].iloc[train_size:].reset_index(drop=True)

    # create the training and testing data sets with sliding door 
    def create_dataset(dataset, time_step):
        X, y = [], []

        for i in range(time_step, len(dataset)):
            X.append(dataset[i-time_step:i, :])
            y.append(dataset[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

        # # Pad with existing values if the lengths are still not equal
        # if len(X) < len(dataset):
        #     num_missing = len(dataset) - len(X)
        #     missing_data = dataset[-num_missing:, :]
        #     missing_data = np.tile(missing_data, (1, X.shape[1] // missing_data.shape[1]))
        #     X = np.concatenate((X, missing_data), axis=0)
        #     y = np.concatenate((y, missing_data[:, 0]))

        # Pad with zeros if the lengths are still not equal
        if len(X) < len(dataset):
            num_missing = len(dataset) - len(X)
            missing_data = np.zeros((num_missing, X.shape[1]))
            X = np.concatenate((X, missing_data), axis=0)
            y = np.concatenate((y, np.zeros(num_missing)))

        return X, y

    if len(train_data) >= time_step and len(test_data) >= time_step:
        # create the training and testing data sets with sliding door
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, _ = create_dataset(test_data, time_step)

        # reshape the input data
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    else:
        # handle the case where train_data length is smaller than time_step
        print("Error: the 'train_test_split' value is too high for the current number of samples. Please lower it or adjust the 'time_step' value.")
        sys.exit(0)
    
    # minutes per each model
    time_dist = 60 * minutes_per_model

    # Create the model (solo or ensemble)
    if model_type == 'solos':
        model = AutoSklearnRegressor(
            time_left_for_this_task=time_dist,
            memory_limit = memory_limit,
            ensemble_kwargs = {'ensemble_size': 1}
        )
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)

    elif model_type == 'ensembles':
        model = AutoSklearnRegressor(
            time_left_for_this_task=time_dist,
            memory_limit = memory_limit,
        )

        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
    elif model_type == 'xgboost':
        # Create the DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        # Define the parameters for XGBoost
        params = {
            'eval_metric': 'mae'
        }

        # Train the XGBoost model
        model = xgb.train(params, dtrain)

        # Predict on the test set
        y_pred = model.predict(dtest)

    else: 
        logger(f'model_type not found: {model_type}')
        sys.exit(0)

    # Inverse transform the predictions and actual values
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    # save the model name
    model_file = f'{out_path}/{building_file}_{y_column}_{imputation_method}_{feature_method}_{n_feature}_{updated_n_feature}_{time_step}_{datelevel}'
    model_file = model_file.replace('.csv', '')

    # calculate metrics
    # logger(f'{bldgname}, {y_column}, {imputation_method}, {feature_method}, n_feature: {n_feature}, time_step: {time_step}')
    # logger(model.leaderboard())

    nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test

    rmse = np.sqrt(mean_squared_error(y_test[~nan_mask], y_pred[~nan_mask]))
    # logger('RMSE: %.3f' % rmse)

    mae = mean_absolute_error(y_test[~nan_mask], y_pred[~nan_mask])
    # logger('MAE: %.3f' % mae)

    mape = mean_absolute_percentage_error(y_test[~nan_mask], y_pred[~nan_mask])
    # logger('MAPE: %.3f' % mape)

    # save file
    if save_model_file == True:
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

    # save plot 
    if save_model_plot == True:
        # plot results
        fig, ax = plt.subplots()

        # Plot the actual values
        ax.plot(y_test, label='Actual Values', alpha=0.7)

        # Plot the predictions
        ax.plot(y_pred, label='Forecasted Values', alpha=0.8)

        # Plot the replaced missing values
        y_test[~nan_mask] = np.nan
        
        ax.plot(y_test, label='Predicted Values', alpha=0.75)

        ax.set_title(f'{bldgname} Consumption')
        ax.set_xlabel(f'Time ({datelevel})')
        ax.set_ylabel(y_column.split('_')[-2] + ' (' + y_column.split('_')[-1] + ')')

        ax.legend()
        plt.grid(True)
        plt.savefig(model_file + '.png')
        plt.close(fig) 

    # return results
    return (model_type, bldgname, y_column, imputation_method, feature_method, n_feature, updated_n_feature, time_step, datelevel, rmse, mae, mape, model_file, model_data_path, building_file, selected_features_delimited)
