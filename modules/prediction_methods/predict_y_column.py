import logging_config
import csv
import pickle
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from autosklearn.regression import AutoSklearnRegressor
from modules.logging_methods.main import logger
import xgboost as xgb
from datetime import datetime

from modules.utils.resample_data import resample_data
from modules.utils.detect_data_frequency import detect_data_frequency
from modules.feature_methods.main import feature_engineering

# Define the function to process each combination of parameters
def predict_y_column(args, startDateTime, endDateTime, config, model_data, model, datelevel):
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

    original_datelevel = detect_data_frequency(model_data)

    # This code aggregates and resamples a DataFrame based on given datelevel
    model_data = resample_data(model_data, datelevel, original_datelevel)

    # Save original n_feature value
    updated_n_feature = n_feature

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
    selected_features = selected_features_delimited.split('|')

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

    # split the data into training and testing sets
    train_size = int(len(data_scaled) * train_test_split)
    # train_data = data_scaled[:train_size, :]
    test_data = data_scaled[train_size:, :]
    
    if len(test_data) < time_step:
        # handle the case where train_data length is smaller than time_step
        print("Error: the 'train_test_split' value is too high for the current number of samples. Please lower it or adjust the 'time_step' value.")
        sys.exit(0)
    
    # Get the most recent time_step days from the data
    recent_data = data_scaled[-time_step:, :]
    
    # Initialize an array to store the predicted consumption
    y_pred_list = []

    pred_len = time_step

    for _ in range(pred_len):
        # Take the last time_step days from the test_data to make the prediction
        X_pred = recent_data[-time_step:, :]

        # Reshape the input data
        X_pred = np.reshape(X_pred, (1, X_pred.shape[0] * X_pred.shape[1]))

        if model_type == 'xgboost':
            X_pred = xgb.DMatrix(X_pred)

        # Predict the next day's consumption
        y_pred = model.predict(X_pred)

        # Inverse transform the prediction
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Append the predicted consumption to the list
        y_pred_list.append(y_pred[0, 0])

        # Shift the recent_data window by one hour
        recent_data = np.roll(recent_data, -1, axis=0)
        
    # Print the predicted consumption
    # print(f"Predicted consumption for the next {pred_len} hours:")
    # for i, consumption in enumerate(y_pred_list):
    #     print(f"Hour {i+1}: {consumption} kWh")

    start = datetime.strptime(startDateTime, config['datetime_format']) or model_data['ds'].iloc[0]
    end = datetime.strptime(endDateTime, config['datetime_format']) or model_data['ds'].iloc[-1]

    print(start,end)

    return y_pred_list, start, end