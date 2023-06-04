import sys
from .linear_regression import linear_regression
from .linear_interpolation import linear_interpolation
from .prophet import prophet
from .lstm import lstm

def preprocessing(model_data, preprocessing_method):
    if (preprocessing_method == 'linear_regression'):
        model_data = linear_regression(model_data)

    elif(preprocessing_method == 'linear_interpolation'):
        model_data = linear_interpolation(model_data)  

    elif(preprocessing_method == 'prophet'):
        model_data = prophet(model_data)  

    elif(preprocessing_method == 'lstm'):
        model_data = lstm(model_data)  

    else:
        print(f'preprocessing_method not found: {preprocessing_method}')
        sys.exit(0)

    return model_data