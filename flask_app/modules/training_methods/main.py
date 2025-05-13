import pickle
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from autosklearn.regression import AutoSklearnRegressor
import xgboost as xgb

from modules.utils.resample_data import resample_data
from modules.utils.detect_data_frequency import detect_data_frequency
from modules.feature_methods.main import feature_engineering
from modules.training_methods.plot_results import plot_results
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
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datetime_format = config['datetime_format']
    table = config['table']

    # Check if the file exists
    if not os.path.exists(model_data_path):
        logger("File not found. Please set save_preprocessed_files: True and run with --preprocess")
        sys.exit()

    # Load model_data separately within each task
    with open(model_data_path, 'rb') as file:
        model_data = pickle.load(file)

    # Group the dataframe by building name and timestamp
    model_data = model_data.set_index('ds')

    # Filter the dataframe to include data within the startDateTime and endDateTime
    if startDateTime and endDateTime:
        # Convert startDateTime and endDateTime to datetime objects
        start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
        end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
        model_data = model_data.loc[(model_data.index >= start_datetime_obj) & (model_data.index <= end_datetime_obj)]
    elif startDateTime:
        # Convert startDateTime to datetime object
        start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
        model_data = model_data.loc[model_data.index >= start_datetime_obj]
    elif endDateTime:
        # Convert endDateTime to datetime object
        end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
        model_data = model_data.loc[model_data.index <= end_datetime_obj]

    future_condition = not endDateTime and save_model_file == True

    # Comment out if you would like to view selected features without datetimes
    if future_condition:
        saved_n_feature = n_feature
        saved_updated_n_feature = updated_n_feature
        n_feature = 0
        updated_n_feature = 0

    model_data = model_data.reset_index()

    out_path = f'{path}/models/{table}_{model_type}'

    original_datelevel = detect_data_frequency(model_data)

    # This code aggregates and resamples a DataFrame based on given datelevel
    model_data = resample_data(model_data, datelevel, original_datelevel)

    # normalize the data, save orginal data column for graphing later
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))

    # check if selected_features are already saved
    selected_features = selected_features_delimited.split('|')
    
    # identify most important features and eliminate less important features    
    if selected_features[0] == '' and not save_model_file:
        # normalize additional features
        add_data_scaled = np.empty((model_data.shape[0], 0))

        for feature in add_feature:
            feature_scaler = StandardScaler()
            add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
            add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

        selected_features = feature_engineering(feature_method, n_fold, add_data_scaled, data_scaled, add_feature)
        selected_features_delimited = '|'.join(selected_features)
        updated_n_feature = min(n_feature, len(selected_features))

    if len(selected_features) > 0 and selected_features[0] != '' and n_feature > 0:
        # normalize selected features
        add_data_scaled = np.empty((model_data.shape[0], 0))

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
        selected_features_delimited = ''
    
    # split the data into training and testing sets
    if save_model_file == True and save_model_plot == False:
        train_size = int(len(data_scaled) * train_test_split)
        train_data = data_scaled
        test_data = data_scaled

        # save y values for benchmarking/plotting
        y_test = model_data['y'].reset_index(drop=True)
        saved_y_test = model_data['y_saved'].reset_index(drop=True)

    else:
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

        # Pad with zeros if the lengths are still not equal
        # if len(X) < len(dataset):
        #     num_missing = len(dataset) - len(X)
        #     missing_data = np.zeros((num_missing, X.shape[1]))
        #     X = np.concatenate((X, missing_data), axis=0)
        #     y = np.concatenate((y, np.zeros(num_missing)))

        return X, y

    if len(train_data) >= time_step and len(test_data) >= time_step:
        # create the training and testing data sets with sliding door
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, _ = create_dataset(test_data, time_step)

        # reshape the input data
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
  
        # if y_train is all zeros, add a tiny noise to make it continuous
        if np.unique(y_train).size < 3:
            y_train = y_train + np.random.normal(0, 1e-6, size=y_train.shape)
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
            include={'feature_preprocessor': ['no_preprocessing']},
            ensemble_kwargs = {'ensemble_size': 1}
        )
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)

    elif model_type == 'ensembles':
        model = AutoSklearnRegressor(
            time_left_for_this_task=time_dist,
            include={'feature_preprocessor': ['no_preprocessing']},
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

    # print(building_file)
    # print(y_pred[0,0])
    # print(y_pred)
    
    y_pred = np.insert(y_pred, 0, y_test.iloc[0])
    y_pred = y_pred[:-1]

    # shape fix (to match y_pred with y_test)
    if len(y_pred) < len(y_test):
        y_test = y_test.iloc[:len(y_pred)].reset_index(drop=True)
        saved_y_test = saved_y_test.iloc[:len(y_pred)].reset_index(drop=True)

    # Comment out if you would like to view selected features without datetimes
    if future_condition:
        n_feature = saved_n_feature
        updated_n_feature = saved_updated_n_feature

    # save the model name
    model_file = f'{out_path}/{building_file}_{y_column}_{imputation_method}_{feature_method}_{n_feature}_{updated_n_feature}_{time_step}_{datelevel}'
    model_file = model_file.replace('.csv', '')

    # calculate metrics
    # logger(f'{bldgname}, {y_column}, {imputation_method}, {feature_method}, n_feature: {n_feature}, time_step: {time_step}')
    # logger(model.leaderboard())

    nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test
    if len(nan_mask) > len(y_pred):
        nan_mask = nan_mask[:len(y_pred)]

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
        plot_results(y_test, y_pred, nan_mask, bldgname, y_column, datelevel, model_file)

    # return results
    return (model_type, bldgname, y_column, imputation_method, feature_method, n_feature, updated_n_feature, time_step, datelevel, rmse, mae, mape, model_file, model_data_path, building_file, selected_features_delimited)
