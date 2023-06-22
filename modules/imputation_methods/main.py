import sys
from .linear_regression import linear_regression
from .linear_interpolation import linear_interpolation
from .prophet import prophet
from .lstm import lstm
from modules.logging_methods.main import logger

def imputation(model_data, imputation_method):
    if (imputation_method == 'linear_regression'):
        model_data = linear_regression(model_data)

    elif(imputation_method == 'linear_interpolation'):
        model_data = linear_interpolation(model_data)  

    elif(imputation_method == 'prophet'):
        model_data = prophet(model_data)  

    elif(imputation_method == 'lstm'):
        model_data = lstm(model_data)  

    else:
        logger(f'imputation_method not found: {imputation_method}')
        sys.exit(0)

    return model_data