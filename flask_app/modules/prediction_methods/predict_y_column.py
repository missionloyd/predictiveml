# import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from datetime import datetime

from modules.utils.resample_data import resample_data
from modules.utils.detect_data_frequency import detect_data_frequency

# from modules.logging_methods.main import logger
# from modules.feature_methods.main import feature_engineering

# from autosklearn.regression import AutoSklearnRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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
    startDateTime = config['startDateTime']
    endDateTime = config['endDateTime']
    datetime_format = config['datetime_format']

    # Checking if startDateTime is a non-empty string
    if len(startDateTime) > 0:
        start = datetime.strptime(startDateTime, config['datetime_format'])
    else:
        start = model_data['ds'].iloc[0]

    # Checking if endDateTime is a non-empty string
    if len(endDateTime) > 0:
        end = datetime.strptime(endDateTime, config['datetime_format'])
    else:
        end = model_data['ds'].iloc[-1]

    # Group the dataframe by building name and timestamp
    model_data_copy = model_data.copy()
    original_datelevel = detect_data_frequency(model_data)
    model_data = model_data.set_index('ds')

    if startDateTime:
        start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
        start_index = model_data.index.searchsorted(start_datetime_obj)
    else:
        start_index = 0

    if endDateTime:
        end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
        end_index = model_data.index.searchsorted(end_datetime_obj, side='right')
    else:
        end_index = len(model_data.index)

    # Correct slicing with time_step adjustment
    model_data = model_data.iloc[-(end_index - start_index + time_step):, :]

    # Apply greater than/less than filtering
    if startDateTime and endDateTime:
        model_data = model_data.loc[(model_data.index >= start_datetime_obj) & (model_data.index > end_datetime_obj)]
    elif startDateTime:
        model_data = model_data.loc[model_data.index >= start_datetime_obj]
    elif endDateTime:
        model_data = model_data.loc[model_data.index > end_datetime_obj]

    # Update start and end indexes after filtering
    if startDateTime:
        start_index = model_data.index.searchsorted(start_datetime_obj)
    if endDateTime:
        end_index = model_data.index.searchsorted(end_datetime_obj, side='right')

    future_condition = not endDateTime and save_model_file == True

    # Comment out if you would like to view selected features without datetimes
    if future_condition:
        n_feature = 0
        updated_n_feature = 0

    model_data = model_data.reset_index()
    
    original_datelevel = detect_data_frequency(model_data)
    # This code aggregates and resamples a DataFrame based on given datelevel
    model_data = resample_data(model_data, datelevel, original_datelevel)

    if len(model_data) <= time_step:
        model_data = model_data_copy
        model_data = resample_data(model_data, datelevel, original_datelevel)
        
        model_data = model_data.iloc[-(end_index - start_index + (time_step * 2)):, :]

    # normalize the data, save orginal data column for graphing later
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))

    # check if selected_features are already saved
    selected_features = selected_features_delimited.split('|')

    # normalize selected features
    add_data_scaled = np.empty((model_data.shape[0], 0))

    if len(selected_features) > 0 and selected_features[0] != '' and n_feature > 0:
        for feature in selected_features:
            feature_scaler = StandardScaler()
            add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
            add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

        # train PCA (Linear Dimensionality Reduction) with multi-feature output
        pca = PCA(n_components=updated_n_feature)
        pca_data = pca.fit_transform(add_data_scaled)
        data_scaled = np.concatenate((data_scaled, pca_data), axis=1)

    # create the training and testing data sets with sliding door 
    def create_dataset(dataset, time_step):
        X, y = [], []

        for i in range(time_step, len(dataset)):
            X.append(dataset[i-time_step:i, :])
            y.append(dataset[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

        # # Pad with zeros if the lengths are still not equal
        # if len(X) < len(dataset):
        #     num_missing = len(dataset) - len(X)
        #     missing_data = np.zeros((num_missing, X.shape[1]))
        #     X = np.concatenate((X, missing_data), axis=0)
        #     y = np.concatenate((y, np.zeros(num_missing)))

        return X, y

    # Initialize an array to store the predicted consumption
    y_pred_list = []

    if future_condition:
        X_test, _ = create_dataset(data_scaled, time_step)

        # if not future_condition:
        #     X_test = X_test[time_step-1:]

        rolling_window = X_test[-time_step-1:-time_step, :]

        for _ in range(time_step):
            # reshape the input data
            
            if model_type == 'xgboost':
                X_test = xgb.DMatrix(rolling_window)

                # Predict the next day's consumption
                y_pred = model.predict(X_test)

            else:
                y_pred = model.predict(rolling_window)

            # Inverse transform the prediction
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

            # Append the predicted consumption to the list
            if y_pred[0, 0] < 0.0:
                y_pred_list.append(np.float32(0.0))
            else:
                y_pred_list.append(y_pred[0, 0])
                
            # update rolling_window
            rolling_window = np.append(rolling_window, y_pred)  # Append the prediction
            rolling_window = rolling_window.reshape(1, -1)  # Reshape to maintain window size
            rolling_window = rolling_window[:, 1:]  # Remove the first element
    else:
        recent_data, _ = create_dataset(data_scaled, time_step)
        recent_data = recent_data
        X_test = recent_data

        for _ in range(time_step):
            # Take the last time_step days from the test_data to make the prediction
            X_test = recent_data

            if model_type == 'xgboost':
                X_test = xgb.DMatrix(X_test)

            # Predict the next day's consumption
            y_pred = model.predict(X_test)

            # Inverse transform the prediction
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

            # Append the predicted consumption to the list
            if y_pred[0, 0] < 0.0:
                y_pred_list.append(np.float32(0.0))
            else:
                y_pred_list.append(y_pred[0, 0])

            recent_data = np.roll(recent_data, -1, axis=0)
            recent_data = recent_data[:-1]

    return y_pred_list, start, end