import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
from tqdm.std import tqdm

from get_weather_data_at_location_and_hour import *

def build_extended_datafile( csv_name ):

    # Import CSV file
    file = f'../clean_data/{csv_name}.csv'
    df = pd.read_csv(file)

    # print(df.head())
    # print(len(df))

    tf = TimezoneFinder()

    df_extended = pd.DataFrame()

    running_chunk = False
    row_counter = 0
    get_elevation = True # Only get the elevation once per file (to improve speed)
    file_invalid = False

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Convert timestamp to local time
        dt_local = datetime.strptime(row['ts'], '%Y-%m-%dT%H:%M:%S')
        local_time_str = dt_local.strftime('%Y-%m-%dT%H:%M')

        # Figure out local time zone
        # Check if the column 'longitude' and 'latitude' has valid values
        if np.isnan(row['longitude']) or np.isnan(row['latitude']):
            # skip processing, break from loop
            file_invalid = True
            break
        else:
            local_time_zone_str = tf.timezone_at(lng=row['longitude'], lat=row['latitude'])
            utc_time_str, timezone_diff_hrs = local_time_to_utc_str(local_time_str, local_time_zone_str)

        dt_utc = datetime.strptime(utc_time_str, '%Y-%m-%dT%H:%M')

        # Check if dt_utc is a new day, and if so, start a new 24-hour chunk
        if dt_utc.hour == 0:
            running_chunk = True
            row_counter = 0

            row_df_24 = pd.DataFrame()  # Initialize the DataFrame for the 24-hour chunk

            # Get the elevation at the given latitude and longitude
            if get_elevation:
                elevation_meters = get_elevation_online(row['latitude'], row['longitude'])
                get_elevation = False

            # Get weather data in 24-hour chunks
            get_24hour_data = True  # Get weather data for the entire UTC 24-hour day
            weather_data_dict_24 = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'], elevation_meters, utc_time_str, timezone_diff_hrs, get_24hour_data)

            # Convert the weather_data_dict to a pandas DataFrame object
            weather_data_df_24 = pd.DataFrame(weather_data_dict_24)

        # Run this row by row until the 24-hour chunk is complete
        if running_chunk:
            row_counter += 1

            # Assemble rows for the 24-hour chunk
            row_df_24 = pd.concat([row_df_24, row.to_frame().T], axis=0)  # to_frame().T converts the Series to a DataFrame row

        # Once the 24-hour chunk is complete, append it to the new DataFrame
        if row_counter == 24:
            running_chunk = False
            row_counter = 0
            
            weather_data_df_24.index = row_df_24.index
            row_df_24 = pd.concat([row_df_24, weather_data_df_24], axis=1)

            # Append the combined_row to the new DataFrame
            df_extended = pd.concat([df_extended, row_df_24], axis=0)
        elif not running_chunk:
            # Get the elevation at the given latitude and longitude
            if get_elevation:
                elevation_meters = get_elevation_online(row['latitude'], row['longitude'])
                get_elevation = False

            # Get weather data
            get_24hour_data = False  # Get weather data for the specified local hour
            weather_data_dict = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'], elevation_meters, utc_time_str, timezone_diff_hrs, get_24hour_data)

            # Convert the weather_data_dict to a pandas Series object
            weather_data_series = pd.Series(weather_data_dict)

            # Append the weather_data_series to the original row
            combined_series = pd.concat([row, weather_data_series], axis=0)

            # Append the combined_row to the new DataFrame
            df_extended = pd.concat([df_extended, combined_series.to_frame().T], axis=0)  # to_frame().T converts the Series to a DataFrame row

    if file_invalid:
        print(f':: ** File {csv_name}_Extended.csv cannot be processed due to invalid entries...')
    else:
        df_extended.columns = df_extended.columns.str.lower()
        #print(df_extended)

        df_extended = df_extended.drop(columns=['forecast', 'time_utc', 'time_local'])

        # Save the DataFrame to a CSV file
        df_extended.to_csv(f'../clean_data_extended/{csv_name}_Extended.csv', index=False)
