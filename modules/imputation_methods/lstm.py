from modules.logging_methods.main import logger
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import sys
pd.options.mode.chained_assignment = None
tf.keras.utils.disable_interactive_logging()

# Move model compilation and training outside of lstm function
def compile_and_train_model(model, X_train, y_train, epochs):
    model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
    early_stopping = EarlyStopping(monitor='loss', patience=epochs)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[early_stopping])

# Train the model on the entire model_data
def create_model_data(model_data, time_step):
    X, y = [], []
    for i in range(time_step, len(model_data)):
        X.append(model_data[i-time_step:i, 0])
        y.append(model_data[i, 0])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    return X, y

def lstm(model_data):
    # hyperparameters
    time_step = 24
    epochs = 5

    # Fill in missing values temporarily
    y_int = model_data['y'].interpolate(method='linear', limit_direction='both')
    y_clean = model_data['y'].dropna()

    # Find the unique indices of missing values in model_data['y']
    # missing_indices = np.unique(model_data['y'].index[model_data['y'].isnull()])

    missing_gaps = []
    group = []

    for i, value in enumerate(model_data['y']):
        if np.isnan(value):
            group.append(i)
        elif len(group) > 0:
            missing_gaps.append(group)
            group = []

    if len(group) > 0:
        missing_gaps.append(group)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(y_clean.values.reshape(-1, 1))

    X_train, y_train = create_model_data(data_scaled, time_step)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    compile_and_train_model(model, X_train, y_train, epochs)  # Compile and train the model

    # Generate predictions for each gap and replace in model_data['y']
    for gap in missing_gaps:
        if gap[0] == 0 or gap[-1] == len(model_data['y']) - 1:
            for idx in gap:
                model_data['y'].iloc[idx] = np.float32(y_int.iloc[idx])
            continue

        if gap[0] - time_step >= 0:  # Check if there are enough data points before the gap
            input_data = model_data['y'].iloc[gap[0] - time_step:gap[0]].values.reshape(-1, 1)
            input_data_scaled = scaler.transform(input_data)
            input_data_scaled = np.reshape(input_data_scaled, (1, time_step, 1))

            for idx in gap:
                input_data = model_data['y'].iloc[idx - time_step:idx].values.reshape(-1, 1)
                input_data_scaled = scaler.transform(input_data)
                input_data_scaled = np.reshape(input_data_scaled, (1, time_step, 1))

                prediction_scaled = model.predict(input_data_scaled)
                prediction = scaler.inverse_transform(prediction_scaled)[0, 0]
                model_data['y'].iloc[idx] = np.float32(prediction)

                # Update input_data_scaled for the next iteration
                input_data_scaled = np.concatenate((input_data_scaled[:, 1:, :], prediction_scaled.reshape(1, 1, 1)), axis=1)

        else:
            for idx in gap:
                # Use the actual value at the index if there are not enough previous data points
                model_data['y'].iloc[idx] = np.float32(y_int.iloc[idx])  # Convert to float32

    # Check for any more missing values and address them
    missing_indices = np.unique(model_data['y'].index[model_data['y'].isnull()])

    if len(missing_indices) > 0:
        print(missing_indices)
        print('Unable to successfully fill in missing gaps using LSTM.')
        sys.exit()

    return model_data