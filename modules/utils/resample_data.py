import pandas as pd
import numpy as np
import sys
from scipy import stats

# This code aggregates and resamples a DataFrame based on the given datelevel
def resample_data(model_data, datelevel, original_datelevel):
    if datelevel != original_datelevel:
        # Convert 'ds' column to pandas datetime format
        model_data['ds'] = pd.to_datetime(model_data['ds'])

        # Save the first and last rows of the original data for comparison
        first_row_original = model_data['ds'].iloc[0]
        last_row_original = model_data['ds'].iloc[-1]

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

        # Save the first and last rows of the aggregated data for comparison
        first_row_aggregated = aggregated_data['ds'].iloc[0]
        last_row_aggregated = aggregated_data['ds'].iloc[-1]

        if last_row_original != last_row_aggregated and len(aggregated_data) > 4:
            aggregated_data = aggregated_data.iloc[:-1]

        # if first_row_original != first_row_aggregated and len(aggregated_data) > 4:
        #     aggregated_data = aggregated_data.iloc[1:]

        model_data = aggregated_data
        # print(model_data)

    return model_data
