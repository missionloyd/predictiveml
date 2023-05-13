import csv, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from autosklearn.regression import AutoSklearnRegressor
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

from prophet import Prophet

# Settings
# PATH = '.' # if running locally
PATH = '/home/lmacy1/predictiveml' # if running on ARCC
data_path = f'{PATH}/clean_data_extended'
buildings_list = ['Stadium_Data_Extended.csv']
save_model_file = False
save_model_plot = False
min_number_of_days = 365
memory_limit = 102400
exclude_column = 'present_co2_tons'

y_columns = ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr', 'present_co2_tons']
add_features = ['temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg']

# Training scope
models = {}
model_types = ['ensembles', 'solos']
preprocessing_methods = ['linear regression', 'linear interpolation', 'prophet']
feature_modes = ['rfe', 'lasso']

# Hyperparameters
n_features_list = list(range(1, len(add_features)))
n_folds = 5
time_steps = [1, 8, 12, 24]
minutes_per_model = 2
split_rate = 0.8

for model_type in model_types:
    out_path = f'{PATH}/models/{model_type}/'

    for building in buildings_list:
        df = pd.read_csv(f'{data_path}/{building}')

        # Convert the data into a Pandas dataframe
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.drop_duplicates(subset=['bldgname', 'ts'])
        df = df.sort_values(['bldgname', 'ts'])

        # Group the dataframe by building name and timestamp
        groups = df.groupby('bldgname')
        df = df.set_index('ts')
        header = ['ts'] + y_columns + add_features

        print(building)

        # cycle through building names if more than one building per file
        for name, group in groups:
            bldgname = name
            group = group.drop_duplicates(subset=['ts'])

            # cycle through commodities
            for y in y_columns:

                col_data = group[header]

                # check if column contains the min number of days and is a valid commodity to train on
                if col_data[y].count() >= min_number_of_days * 24 and y != exclude_column:

                    # cycle through preprocessing methods
                    for preprocessing_method in preprocessing_methods:

                        model_data = col_data.copy()
                        model_data = model_data.rename(columns={ y: 'y', 'ts': 'ds' })
                        model_data = model_data.sort_values(['ds'])

                        # save the original values into new column
                        model_data['y_saved'] = model_data['y']

                        # Fill in missing values (Preprocessing)

                        # *** Linear Regression (1/5) ***
                        if (preprocessing_method == 'linear regression'):
                            m = LinearRegression()

                            X_train = model_data[model_data['y'].notna()]['ds'].values.reshape(-1, 1)
                            y_train = model_data[model_data['y'].notna()]['y'].values.reshape(-1, 1)
                            m.fit(X_train, y_train)

                            X_test = model_data[model_data['y'].isna()]['ds'].values.reshape(-1, 1)
                            X_test = X_test.astype(np.float32)

                            y_pred = m.predict(X_test)

                            model_data.loc[model_data['y'].isna(), 'y'] = y_pred.flatten()


                        # *** Linear Interpolation (2/5) ***
                        elif(preprocessing_method == 'linear interpolation'):
                            model_data['y'] = model_data['y'].interpolate(method='linear', limit_direction='both')    
                        

                        # *** Cubic Interpolation (3/5) ***
                        elif(preprocessing_method == 'cubic interpolation'):
                            model_data['y'] = model_data['y'].interpolate(method='cubic', limit_direction='both')


                        # *** Prophet (4/5) ***
                        elif(preprocessing_method == 'prophet'):
                            m = Prophet()
                            m.fit(model_data)
                            future = m.make_future_dataframe(periods=0, freq='H')
                            forecast = m.predict(future)
                            model_data['y'] = model_data['y'].fillna(forecast['yhat'])
                        
                        # *** ARIMA (daily seasonality, m=24) (5/5) ***
                        # elif(preprocessing_method == 'arima'):
                            # m = auto_arima(model_data[model_data['y'].notna()]['y'].values, seasonal=True, m=24*365, suppress_warnings=False)

                            # y_pred, conf_int = m.predict(n_periods=model_data['y'].isna().sum())
                            # model_data.loc[model_data['y'].isna(), 'y'] = y_pred

                        for feature_mode in feature_modes:
                            for n_features in n_features_list:

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
                                if feature_mode == 'rfe':
                                    # RFE feature selection with linear regression estimator
                                    lin_reg = LinearRegression()
                                    rfe = RFECV(estimator=lin_reg, cv=n_folds)
                                    rfe.fit(add_data_scaled, data_scaled)

                                    selected_features = np.array(add_features)[rfe.support_]
                                    print(selected_features)

                                elif feature_mode == 'lasso':
                                    # LassoCV feature selection with cross-validation
                                    lassocv = LassoCV(cv=n_folds)
                                    lassocv.fit(add_data_scaled, data_scaled)

                                    selected_features = np.array(add_features)[lassocv.coef_ != 0]
                                    print(selected_features)

                                # normalize selected features
                                add_data_scaled = np.empty((model_data.shape[0], 0))

                                for feature in selected_features:
                                    feature_scaler = StandardScaler()
                                    add_feature_scaled = feature_scaler.fit_transform(model_data[feature].values.reshape(-1, 1))
                                    add_data_scaled = np.concatenate((add_data_scaled, add_feature_scaled), axis=1)

                                # handle case where n_features is greater than or equal to selected features
                                n_components = n_features
                                if (n_features >= add_data_scaled.shape[1]):
                                    n_components = add_data_scaled.shape[1]

                                # train PCA (Linear Dimensionality Reduction) with multi feature output
                                pca = PCA(n_components=n_components)
                                pca_data = pca.fit_transform(add_data_scaled)
                                data_scaled = np.concatenate((data_scaled, pca_data), axis=1)
                                
                                # split the data into training and testing sets
                                train_size = int(len(data_scaled) * split_rate)
                                test_size = len(data_scaled) - train_size
                                train_data = data_scaled[0:train_size,:]
                                test_data = data_scaled[train_size:len(data_scaled),:]
                                saved_test_data = saved_data_scaled[train_size:len(data_scaled),:]

                                # cycle through time steps
                                for time_step in time_steps:

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
                                    else:
                                        model = AutoSklearnRegressor(
                                            time_left_for_this_task=time_dist,
                                            memory_limit = memory_limit,
                                        )

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
                                    model_file = f'{out_path}/{bldgname}_{y}_{preprocessing_method}_{feature_mode}_{n_features}_{time_step}'
                                    model_file = model_file.replace(' ', '-').lower()

                                    # calculate metrics
                                    print(f'{bldgname}, {y}, Time Step: {time_step}')
                                    print(model.leaderboard())

                                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                    print('RMSE: %.3f' % rmse)

                                    mae = mean_absolute_error(y_test, y_pred)
                                    print('MAE: %.3f' % mae)

                                    r2 = r2_score(y_test, y_pred)
                                    print('R2: %.3f' % r2)

                                    # save results
                                    models[(model_type, bldgname, y, preprocessing_method, feature_mode, n_features, time_step)] = (rmse, mae, r2, model_file)

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
                                        nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test
                                        y_test[~nan_mask] = np.nan
                                        
                                        ax.plot(y_test, label='Predicted Values', alpha=0.75)

                                        ax.set_title(f'{bldgname} Consumption')
                                        ax.set_xlabel('Time (Hours)')
                                        ax.set_ylabel(y.split('_')[-2] + ' (' + y.split('_')[-1] + ')')

                                        ax.legend()
                                        plt.grid(True)
                                        plt.savefig(model_file + '.png')
                                        plt.close(fig)         


# Create a CSV files to save the results
header = ['model_type','bldgname', 'y', 'preprocessing_method', 'feature_mode', 'n_features', 'time_step', 'rmse', 'mae', 'r2', 'model_file']
rows = []

# create csv file for each model folder
for m_type in model_types:
    out_path = f'{PATH}/models/{m_type}'

    with open(f'{out_path}/results.csv', mode='w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(header)

        for name, (rmse, mae, r2, model_file) in models.items():
            model_type, bldgname, y, preprocessing_method, feature_mode, n_features, time_step = name

            # Write the row to the CSV file
            if m_type == model_type:
                row = [model_type, bldgname, y, preprocessing_method, feature_mode, n_features, time_step, rmse, mae, r2, model_file + '.pkl']
                writer.writerow(row)
                rows.append(row)

# create master results csv
with open(f'{PATH}/results.csv', mode='w') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(header)

    for row in rows:
        writer.writerow(row)
