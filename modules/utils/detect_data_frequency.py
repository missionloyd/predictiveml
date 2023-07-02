import pandas as pd

def detect_data_frequency(data):
    # Check the time differences between consecutive timestamps
    time_diffs = data['ds'].diff()

    # Identify the unique time differences
    unique_diffs = time_diffs.unique()

    # Count the occurrences of each time difference
    diff_counts = time_diffs.value_counts()

    # Determine the most common time difference (frequency)
    most_common_diff = diff_counts.idxmax()

    # Map the most common time difference to the corresponding frequency alias
    frequency_mapping = {
        pd.Timedelta(minutes=1): '1min',
        pd.Timedelta(minutes=5): '5min',
        pd.Timedelta(minutes=10): '10min',
        pd.Timedelta(minutes=15): '15min',
        pd.Timedelta(minutes=30): '30min',
        pd.Timedelta(hours=1): 'hour',
        pd.Timedelta(days=1): 'day',
        pd.Timedelta(days=30): 'month',
        pd.Timedelta(days=365): 'year'
    }

    # Set the datelevel based on the most common time difference
    original_datelevel = frequency_mapping.get(most_common_diff, '1min')  # Default to '1min' if frequency is not recognized

    return original_datelevel
