import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(model_data):
    if model_data['y'].isna().sum() == 0:
        return model_data

    m = LinearRegression()

    X_train = model_data[model_data['y'].notna()]['ds'].values.reshape(-1, 1)
    y_train = model_data[model_data['y'].notna()]['y'].values.reshape(-1, 1)
    m.fit(X_train, y_train)

    X_test = model_data[model_data['y'].isna()]['ds'].values.reshape(-1, 1)
    X_test = X_test.astype(np.float32)

    y_pred = m.predict(X_test)

    model_data.loc[model_data['y'].isna(), 'y'] = y_pred.flatten()

    return model_data
