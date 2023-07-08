import csv, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from autosklearn.regression import AutoSklearnRegressor

from fbprophet import Prophet

buildings_list = ['Energy_Innovation_Center_Data.csv', 'Stadium_Data.csv']

in_path = './clean_data/'

# types of models
models = {}
model_types = ['ensembles', 'solos']

time_steps = [1, 12, 24, 36, 72, 168, 288]

for model_type in model_types:
    out_path = f'./models/{model_type}/'

    for building in buildings_list:
        df = pd.read_csv(in_path + building)

        # Convert the data into a Pandas dataframe
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.drop_duplicates(subset=['bldgname', 'ts'])
        df = df.sort_values(['bldgname', 'ts'])

        # Group the dataframe by building name and timestamp
        groups = df.groupby('bldgname')
        df = df.set_index('ts')

        orig_cols = df.columns
        y_columns = ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr', 'present_co2_tons']
        header = ['ts'] + y_columns

        print(building)

        for name, group in groups:
            bldgname = name

            group = group.drop_duplicates(subset=['ts'])

            for y in y_columns:
                model_data = group[header]
                if model_data[y].count() >= 365*24 and y != 'present_co2_tons':

                    model_data = model_data.rename(columns={ y: 'y', 'ts': 'ds' })
                    model_data = model_data.sort_values(['ds'])

                    # save the original values into new column
                    model_data['y_saved'] = model_data['y']

                    # Interpolate missing values in 'y' column of model_data using linear method and fill values in both directions.
                    # model_data['y'] = model_data['y'].interpolate(method='linear', limit_direction='both')    

                    # Fill in missing values using Prophet
                    m = Prophet()
                    m.fit(model_data)
                    future = m.make_future_dataframe(periods=0, freq='H')
                    forecast = m.predict(future)
                    model_data['y'] = model_data['y'].fillna(forecast['yhat'])

                    # normalize the data, save orginal data column for graphing later
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data_scaled = scaler.fit_transform(model_data['y'].values.reshape(-1, 1))
                    saved_data_scaled = scaler.fit_transform(model_data['y_saved'].values.reshape(-1, 1))

                    # split the data into training and testing sets
                    train_size = int(len(data_scaled) * 0.8)
                    test_size = len(data_scaled) - train_size
                    train_data = data_scaled[0:train_size,:]
                    test_data = data_scaled[train_size:len(data_scaled),:]
                    saved_test_data = saved_data_scaled[train_size:len(data_scaled),:]

                    for time_step in time_steps:
                        # define the window size
                        time_step = time_step

                        # create the training data set
                        def create_dataset(dataset, time_step):
                            X, y = [], []
                            for i in range(time_step, len(dataset)):
                                X.append(dataset[i-time_step:i, 0])
                                y.append(dataset[i, 0])
                            X, y = np.array(X), np.array(y)
                            return X, y

                        X_train, y_train = create_dataset(train_data, time_step)

                        # create the testing data set
                        X_test, y_test = create_dataset(test_data, time_step)
                        saved_X_test, saved_y_test = create_dataset(saved_test_data, time_step)

                        # reshape the input data
                        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
                        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
                        saved_X_test = np.reshape(saved_X_test, (saved_X_test.shape[0], saved_X_test.shape[1]))

                        # 2 hours each task
                        time_dist = 60*5

                        # Create the model (solo or ensemble)
                        if model_type == 'solos':
                            model = AutoSklearnRegressor(
                                time_left_for_this_task=time_dist,
                                per_run_time_limit=int(time_dist/10),
                                memory_limit = 102400,
                                ensemble_kwargs = {'ensemble_size': 1}
                            )
                        else:
                            model = AutoSklearnRegressor(
                                time_left_for_this_task=time_dist,
                                per_run_time_limit=int(time_dist/10),
                                memory_limit = 102400,
                            )


                        # Train the model
                        model.fit(X_train, y_train)
                        
                        # Predict on the test set
                        y_pred = model.predict(X_test)

                        # # Inverse transform the predictions and actual values
                        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
                        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
                        saved_y_test = scaler.inverse_transform(saved_y_test.reshape(-1, 1))
                        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

                        # save the model
                        model_file = f'{out_path}{bldgname}_{y}_{time_step}_model'

                        # with open(model_file + '.pkl', 'wb') as file:
                        #     pickle.dump(model, file)

                        # calculate metrics
                        print(f'{bldgname}, {y}, Time Step: {time_step}')
                        # print(test_size, len(y_test))
                        print(model.leaderboard())

                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        print('RMSE: %.3f' % rmse)

                        mae = mean_absolute_error(y_test, y_pred)
                        print('MAE: %.3f' % mae)

                        r2 = r2_score(y_test, y_pred)
                        print('R2: %.3f' % r2)

                        # save results
                        models[(model_type, bldgname, y, time_step)] = (rmse, mae, r2, model_file)

                        # plot results
                        fig, ax = plt.subplots()

                        # Plot the actual values
                        # ax.plot(np.concatenate([y_train, y_test]), label='Actual Values')
                        ax.plot(y_test, label='Actual Values', alpha=0.7)

                        # Plot the predictions
                        # ax.plot(range(train_len, train_len + len(y_test)), y_pred, label='Predicted Values')
                        ax.plot(y_pred, label='Forecasted Values', alpha=0.8)

                        # Plot the replaced missing values
                        nan_mask = np.isnan(saved_y_test)  # boolean mask of NaN values in saved_y_test
                        y_test[~nan_mask] = np.nan
                        
                        ax.plot(y_test, label='Predicted Values', alpha=0.75)

                        ax.set_title(f'{bldgname} {y}_{time_step} Consumption')
                        ax.set_xlabel('Time (Hours)')
                        ax.set_ylabel(y.split('_')[-2] + ' (' + y.split('_')[-1] + ')')

                        ax.legend()
                        plt.grid(True)
                        plt.savefig(model_file + '.png')
                        plt.close(fig)


# Create CSV files to save the results

header = ['model_type', 'bldgname', 'y', 'time_step', 'rmse', 'mae', 'r2', 'model_file']
rows = []

# create csv file for each model folder
for m_type in model_types:
    out_path = f'./models/{m_type}/'

    with open(f'{out_path}results.csv', mode='w') as results_file:
        writer = csv.writer(results_file)
        writer.writerow(header)

        # Plot the actual values and predictions
        for name, (rmse, mae, r2, model_file) in models.items():
            model_type, bldgname, y, time_step = name

            # Write the row to the CSV file
            if m_type == model_type:
                row = [model_type, bldgname, y, time_step, rmse, mae, r2, model_file + '.pkl']
                writer.writerow(row)
                rows.append(row)

# create master results csv
with open('results.csv', mode='w') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(header)

    for row in rows:
        writer.writerow(row)

