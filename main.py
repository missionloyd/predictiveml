#!/usr/bin/env python
# coding: utf-8

# In[19]:


import csv
import pickle
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from autosklearn.regression import AutoSklearnRegressor
from joblib import Parallel, delayed
from itertools import product

PATH = '.'
ARCC_PATH = '/home/lmacy1/predictiveml'
sys.path.append(ARCC_PATH) # if running on ARCC

from modules.preprocessing_methods.main import preprocessing
from modules.feature_modes.main import feature_engineering


# In[20]:


# Define the function to process each combination of parameters
def train_model(
    model_type,
    building,
    y_column,
    preprocessing_method,
    feature_mode,
    n_features,
    time_step,
    header,
    data_path,
    add_features,
    min_number_of_days,
    exclude_column,
    n_folds,
    split_rate,
    minutes_per_model,
    memory_limit
):

    out_path = f'{PATH}/models/{model_type}'

    df = pd.read_csv(f'{data_path}/{building}')

    # Convert the data into a Pandas dataframe
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.drop_duplicates(subset=['bldgname', 'ts'])
    df = df.sort_values(['bldgname', 'ts'])

    # Group the dataframe by building name and timestamp
    groups = df.groupby('bldgname')
    df = df.set_index('ts')

    #   print(building)

    # cycle through building names if more than one building per file
    for name, group in groups:
        bldgname = name
        group = group.drop_duplicates(subset=['ts'])

        col_data = group[header]

        # check if column contains the min number of days and is a valid commodity to train on
        if col_data[y_column].count() >= min_number_of_days * 24 and y_column != exclude_column:

            model_data = col_data.copy()
            model_data = model_data.rename(columns={ y_column: 'y', 'ts': 'ds' })
            model_data = model_data.sort_values(['ds'])

            # save the original values into new column
            model_data['y_saved'] = model_data['y']

            # Fill in missing values (preprocessing)
            model_data = preprocessing(model_data, preprocessing_method)

            # normalize the data, save orginal data column for graphing later
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))
            saved_data_scaled = scaler.fit_transform(model_data['y_saved'].values.reshape(-1, 1))

            # normalize additional features
            add_data_scaled = np.empty((model_data.shape[0], 0))

            for feature in add_features:
                feature_scaler = StandardScaler()
                add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
                add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

            # identify most important features and eliminate less important features
            selected_features = feature_engineering(feature_mode, n_folds, add_data_scaled, data_scaled, add_features)

            # normalize selected features
            add_data_scaled = np.empty((model_data.shape[0], 0))

            for feature in selected_features:
                feature_scaler = StandardScaler()
                add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
                add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

            # handle case where n_features is greater than or equal to selected features
            if (n_features >= add_data_scaled.shape[1]):
                n_features = add_data_scaled.shape[1]

            # train PCA (Linear Dimensionality Reduction) with multi feature output
            pca = PCA(n_components=n_features)
            pca_data = pca.fit_transform(add_data_scaled)
            data_scaled = np.concatenate((data_scaled, pca_data), axis=1)
            
            # split the data into training and testing sets
            train_size = int(len(data_scaled) * split_rate)
            test_size = len(data_scaled) - train_size
            train_data = data_scaled[0:train_size,:]
            test_data = data_scaled[train_size:len(data_scaled),:]
            saved_test_data = saved_data_scaled[train_size:len(data_scaled),:]

            # define the window size
            window_size = time_step

            # create the training and testing data sets
            def create_dataset(dataset, window_size):
                X, y = [], []

                for i in range(window_size, len(dataset)):
                    X.append(dataset[i-window_size:i, :])
                    y.append(dataset[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
                return X, y

            X_train, y_train = create_dataset(train_data, window_size)
            X_test, y_test = create_dataset(test_data, window_size)
            saved_X_test, saved_y_test = create_dataset(saved_test_data, window_size)

            # reshape the input data
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            saved_X_test = np.reshape(saved_X_test, (saved_X_test.shape[0], saved_X_test.shape[1]))
            
            # minutes per each model
            time_dist = 60 * minutes_per_model

            # Create the model (solo or ensemble)
            if model_type == 'solos':
                model = AutoSklearnRegressor(
                    time_left_for_this_task=time_dist,
                    memory_limit = memory_limit,
                    ensemble_kwargs = {'ensemble_size': 1}
                )
            elif model_type == 'ensembles':
                model = AutoSklearnRegressor(
                    time_left_for_this_task=time_dist,
                    memory_limit = memory_limit,
                )
            else: 
                print(f'model_type not found: {model_type}')
                sys.exit(0)

            # Train the model
            model.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test)

            # Inverse transform the predictions and actual values
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
            saved_y_test = scaler.inverse_transform(saved_y_test.reshape(-1, 1))
            y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

            # save the model name
            model_file = f'{out_path}/{bldgname}_{y_column}_{preprocessing_method}_{feature_mode}_{n_features}_{time_step}'
            model_file = model_file.replace(' ', '-').lower()

            # calculate metrics
            print(f'{bldgname}, {y_column}, {preprocessing_method}, {feature_mode}, n_features: {n_features}, time_step: {time_step}')
            # print(model.leaderboard())

            nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test

            rmse = np.sqrt(mean_squared_error(y_test[~nan_mask], y_pred[~nan_mask]))
            print('RMSE: %.3f' % rmse)

            mae = mean_absolute_error(y_test[~nan_mask], y_pred[~nan_mask])
            print('MAE: %.3f' % mae)

            r2 = r2_score(y_test[~nan_mask], y_pred[~nan_mask])
            print('R2: %.3f' % r2)
            print('\n')

            # return results
            return (model_type, bldgname, y_column, preprocessing_method, feature_mode, n_features, time_step, rmse, mae, r2, model_file)


# In[ ]:


if __name__ == '__main__':

    # Settings
    data_path = f'{PATH}/clean_data_extended'
    buildings_list = ['Stadium_Data_Extended.csv']
    save_model_file = False
    save_model_plot = False
    min_number_of_days = 365
    memory_limit = 102400
    exclude_column = 'present_co2_tons'
    warnings.filterwarnings("ignore")

    y_columns = ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr', 'present_co2_tons']
    add_features = ['temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg']
    header = ['ts'] + y_columns + add_features

    # Training scope
    models = {}
    model_types = ['ensembles', 'solos']
    preprocessing_methods = ['linear_regression', 'linear_interpolation', 'prophet', 'lstm']
    feature_modes = ['rfecv', 'lassocv']

    # Hyperparameters
    n_features = list(range(1, len(add_features)))
    n_folds = 5
    time_steps = [1, 8, 12, 24]
    minutes_per_model = 2
    split_rate = 0.8

    # Generate a list of arguments for model training
    arguments = list(product(
        model_types,
        buildings_list,
        y_columns,
        preprocessing_methods,
        feature_modes,
        n_features,
        time_steps,
        [header],
        [data_path],
        [add_features],
        [min_number_of_days],
        [exclude_column],
        [n_folds],
        [split_rate],
        [minutes_per_model],
        [memory_limit]
    ))

    # Execute the training function in parallel for each set of arguments
    results = Parallel(n_jobs=-1, prefer="processes")(delayed(train_model)(*arg) for arg in arguments)

    # Create a CSV file to save the results
    results_header = ['model_type', 'bldgname', 'y_column', 'preprocessing_method', 'feature_mode', 'n_features', 'time_step', 'rmse', 'mae', 'r2', 'model_file']

    # Save the results to the CSV file
    with open(f'{PATH}/results.csv', mode='w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(results_header)
        writer.writerows(results)

