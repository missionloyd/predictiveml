import pandas as pd

# This code aggregates and resamples a DataFrame based on the given datelevel
def resample_data(model_data, datelevel, original_datelevel):
    if datelevel != original_datelevel:
        # Convert 'ds' column to pandas datetime format
        model_data['ds'] = pd.to_datetime(model_data['ds'])

        # Define frequency mapping based on datelevel
        frequency_mapping = {
            'hour': 'H',
            'day': 'D',
            'month': 'M',
            'year': 'Y'
        }

        # Get the desired frequency for grouping
        frequency = frequency_mapping[datelevel]

        # Determine the start of the period based on the desired datelevel
        if datelevel == 'hour':
            model_data['ds'] = model_data['ds'].dt.floor('H')
        elif datelevel == 'day':
            model_data['ds'] = model_data['ds'].dt.floor('D')
        elif datelevel == 'month':
            model_data['ds'] = model_data['ds'].dt.to_period('M').dt.start_time
        elif datelevel == 'year':
            model_data['ds'] = model_data['ds'].dt.to_period('Y').dt.start_time

        # Group by the modified 'ds' column and perform aggregation
        aggregated_data = model_data.groupby(pd.Grouper(key='ds', freq=frequency)).sum(numeric_only=True).reset_index()
        model_data = aggregated_data

    return model_data
