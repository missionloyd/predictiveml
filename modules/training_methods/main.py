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

from modules.feature_methods.main import feature_engineering

# Define the function to process each combination of parameters
def train_model(args):
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
    min_number_of_days = args['min_number_of_days']
    exclude_column = args['exclude_column']
    n_fold = args['n_fold']
    split_rate = args['split_rate']
    minutes_per_model = args['minutes_per_model']
    memory_limit = args['memory_limit']
    save_model_file = args['save_model_file']
    save_model_plot = args['save_model_plot']
    path = args['path']
    
    # Load model_data separately within each task
    with open(model_data_path, 'rb') as file:
        model_data = pickle.load(file)

    out_path = f'{path}/models/{model_type}'

    # normalize the data, save orginal data column for graphing later
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))
    saved_data_scaled = scaler.fit_transform(model_data['y_saved'].values.reshape(-1, 1))

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

    for feature in selected_features:
        feature_scaler = StandardScaler()
        add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
        add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

    # handle case where n_features is greater than or equal to selected features
    if (n_feature >= add_data_scaled.shape[1]):
        n_feature = add_data_scaled.shape[1]

    # train PCA (Linear Dimensionality Reduction) with multi feature output
    pca = PCA(n_components=n_feature)
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
    model_file = f'{out_path}/{bldgname}_{y_column}_{imputation_method}_{feature_method}_{n_feature}_{time_step}'
    model_file = model_file.replace(' ', '-').lower()

    # calculate metrics
    print(f'{bldgname}, {y_column}, {imputation_method}, {feature_method}, n_feature: {n_feature}, time_step: {time_step}')
    # print(model.leaderboard())

    nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test

    rmse = np.sqrt(mean_squared_error(y_test[~nan_mask], y_pred[~nan_mask]))
    # print('RMSE: %.3f' % rmse)

    mae = mean_absolute_error(y_test[~nan_mask], y_pred[~nan_mask])
    # print('MAE: %.3f' % mae)

    mape = mean_absolute_percentage_error(y_test[~nan_mask], y_pred[~nan_mask])
    # print('R2: %.3f' % mape)

    # save file
    if save_model_file == True:
        with open(model_file + '.pkl', 'wb') as file:
            pickle.dump(model, file)

    # save plot 
    if save_model_plot == True:
        # plot results
        fig, ax = plt.subplots()

        # Plot the actual values
        ax.plot(y_test, label='Actual Values', alpha=0.7)
        # ax.plot(np.concatenate([y_train, y_test]), label='Actual Values')

        # Plot the predictions
        ax.plot(y_pred, label='Forecasted Values', alpha=0.8)
        # ax.plot(range(train_len, train_len + len(y_test)), y_pred, label='Predicted Values')

        # Plot the replaced missing values
        y_test[~nan_mask] = np.nan
        
        ax.plot(y_test, label='Predicted Values', alpha=0.75)

        ax.set_title(f'{bldgname} Consumption')
        ax.set_xlabel('Time (Hours)')
        ax.set_ylabel(y_column.split('_')[-2] + ' (' + y_column.split('_')[-1] + ')')

        ax.legend()
        plt.grid(True)
        plt.savefig(model_file + '.png')
        plt.close(fig) 

    # return results
    return (model_type, bldgname, y_column, imputation_method, feature_method, n_feature, time_step, rmse, mae, mape, model_file)
