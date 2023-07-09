import pandas as pd
import numpy as np
import sys
from scipy import stats

# This code aggregates and resamples a DataFrame based on the given datelevel
def resample_data(model_data, datelevel, original_datelevel, resample_z_score):
    if datelevel != original_datelevel:
        # Convert 'ds' column to pandas datetime format
        model_data['ds'] = pd.to_datetime(model_data['ds'])

        # Define frequency mapping based on datelevel
        frequency_mapping = {
            'hour': 'H',
            'day': 'D',
            'month': 'MS',
            'year': 'YS'
        }

        # Get the desired frequency for grouping
        frequency = frequency_mapping[datelevel]

        # Determine the start of the period based on the desired datelevel
        if datelevel == 'hour':
            model_data['ds'] = model_data['ds'].dt.floor('H')
        elif datelevel == 'day':
            model_data['ds'] = model_data['ds'].dt.floor('D')
        elif datelevel == 'month':
            model_data['ds'] = model_data['ds'].dt.to_period('M').dt.to_timestamp('M')
        elif datelevel == 'year':
            model_data['ds'] = model_data['ds'].dt.to_period('Y').dt.to_timestamp('Y')

        # Group by the modified 'ds' column and perform aggregation
        aggregated_data = model_data.groupby(pd.Grouper(key='ds', freq=frequency)).sum(numeric_only=True).reset_index()

        # Replace outliers with the mean of non-outlier columns for monthly and yearly datelevels
        if resample_z_score != -1 and datelevel in ['month', 'year']:
            for column in aggregated_data.columns:
                if column != 'ds':
            
                    # column = 'y'
                    # Calculate the mean and standard deviation of the column
                    mean = aggregated_data[column].mean()
                    std = aggregated_data[column].std()

                    # Define the upper and lower bounds for outliers
                    upper_bound = mean + (resample_z_score * std)
                    lower_bound = mean - (resample_z_score * std)

                    # Calculate the mean of non-outlier values
                    non_outlier_mean = aggregated_data.loc[(aggregated_data[column] > lower_bound) & (aggregated_data[column] < upper_bound), column].mean()

                    # Replace outliers with the mean of non-outlier values
                    aggregated_data.loc[(aggregated_data[column] > upper_bound) | (aggregated_data[column] < lower_bound), column] = non_outlier_mean


        model_data = aggregated_data

    print(model_data)

    return model_data
