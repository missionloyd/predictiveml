from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

def lstm(model_data):
    # hyperparameters
    window_size = 24
    epochs = 5

    # Fill in missing values temporarily
    y_int = model_data['y'].interpolate(method='linear', limit_direction='both')
    y_clean = model_data['y'].dropna()

    # Find the unique indices of missing values in model_data['y']
    missing_indices = np.unique(model_data['y'].index[model_data['y'].isnull()])

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

    # print(f'# of gaps: {len(missing_gaps)}')

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(y_clean.values.reshape(-1, 1))

    # Train the model on the entire model_data
    def create_model_data(model_data, window_size):
        X, y = [], []
        for i in range(window_size, len(model_data)):
            X.append(model_data[i-window_size:i, 0])
            y.append(model_data[i, 0])
        X, y = np.array(X), np.array(y)
        return X, y

    X_train, y_train = create_model_data(data_scaled, window_size)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)
    early_stopping = EarlyStopping(monitor='loss', patience=epochs)  # Define early stopping criteria

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, callbacks=[early_stopping])

    # Generate predictions for each gap and replace in model_data['y']
    for gaps in missing_gaps:
        for i in range(0, len(gaps), window_size):
            sub_gaps = gaps[i:i+window_size]
            start_index = sub_gaps[0]
            end_index = sub_gaps[-1]
            gap_length = end_index - start_index + 1

            # Create input sequence for the sub-gap
            X_pred_gap, _ = create_model_data(data_scaled, window_size)
            X_pred_gap = X_pred_gap[start_index-window_size:end_index]
            X_pred_gap = np.reshape(X_pred_gap, (X_pred_gap.shape[0], X_pred_gap.shape[1], 1))

            # Check if the sub-gap is long enough for making predictions or if there are gaps at the start or end of the data
            if X_pred_gap.shape[0] < window_size - 1 or gaps[0] == 0 or gaps[-1] == len(model_data) - 1:
                # Fill in the remaining gaps using interpolation
                model_data.loc[model_data.index[start_index:end_index+1], 'y'] = y_int.iloc[start_index:end_index+1].values

            else:
                # Generate prediction for the sub-gap
                y_pred_gap = model.predict(X_pred_gap)
                y_pred = scaler.inverse_transform(y_pred_gap.reshape(-1, 1))

                # Replace the sub-gap in model_data['y'] with the predicted values
                model_data.loc[model_data.index[start_index:end_index+1], 'y'] = y_pred[:gap_length].reshape(-1)

    return model_data