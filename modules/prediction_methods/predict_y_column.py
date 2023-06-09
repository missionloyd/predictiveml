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

from modules.utils.resample_data import resample_data
from modules.utils.detect_data_frequency import detect_data_frequency
from modules.feature_methods.main import feature_engineering

# Define the function to process each combination of parameters
def predict_y_column(args, model_data, model, datelevel):
    model_data_path = args['model_data_path']
    bldgname = args['bldgname']
    building_file = args['building_file']
    y_column = args['y_column']
    imputation_method = args['imputation_method']
    model_type = args['model_type']
    feature_method = args['feature_method']
    n_feature = args['n_feature']
    time_step = args['time_step']
    header = args['header']
    data_path = args['data_path']
    add_feature = args['add_feature']
    exclude_column = args['exclude_column']
    n_fold = args['n_fold']
    train_test_split = args['train_test_split']
    minutes_per_model = args['minutes_per_model']
    memory_limit = args['memory_limit']
    save_model_file = args['save_model_file']
    save_model_plot = args['save_model_plot']
    path = args['path']

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

    # identify most important features and eliminate less important features
    selected_features = feature_engineering(feature_method, n_fold, add_data_scaled, data_scaled, add_feature)

    # normalize selected features
    add_data_scaled = np.empty((model_data.shape[0], 0))

    if len(selected_features) > 0:
        for feature in selected_features:
            feature_scaler = StandardScaler()
            add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
            add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

        # ensures that updated_n_feature does not exceed the number of selected features or the number of samples in add_data_scaled
        min_nfeatures_nsamples = min(add_data_scaled.shape[0], add_data_scaled.shape[1])
        if updated_n_feature >= min_nfeatures_nsamples:
            updated_n_feature = min_nfeatures_nsamples

        # train PCA (Linear Dimensionality Reduction) with multi-feature output
        pca = PCA(n_components=updated_n_feature)
        pca_data = pca.fit_transform(add_data_scaled)
        data_scaled = np.concatenate((data_scaled, pca_data), axis=1)
    else:
        # Handle the case where no features are selected
        updated_n_feature = 0


    # split the data into training and testing sets
    train_size = int(len(data_scaled) * train_test_split)
    # train_data = data_scaled[:train_size, :]
    test_data = data_scaled[train_size:, :]

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
    
    if len(test_data) >= time_step:
        X_test, _ = create_dataset(test_data, time_step)

        # reshape the input data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    
    else:
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

    start = model_data['ds'].iloc[0]
    end = model_data['ds'].iloc[-1]

    return y_pred_list, start, end