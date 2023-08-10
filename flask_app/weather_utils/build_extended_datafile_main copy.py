import sys
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
from tqdm.std import tqdm

from .get_weather_data_at_location_and_hour import *


def create_extended_datafile_main(csv_name, config):
    clean_data_path = config['clean_data_path']

    # Import CSV file
    file = f'{clean_data_path}/{csv_name}.csv'
    df = pd.read_csv(file)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    tf = TimezoneFinder()

    df_extended = pd.DataFrame()

    running_chunk = False
    row_counter = 0
    get_elevation = True  # Only get the elevation once per file (to improve speed)
    file_invalid = False

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Convert timestamp to local time
        dt_local = datetime.strptime(row['ts'], config['datetime_format'])
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
            # print(timezone_diff_hrs)
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
            # weather_data_dict_24 = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
            #                                                               elevation_meters, utc_time_str,
            #                                                               timezone_diff_hrs, get_24hour_data)
            weather_data_dict_24 = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
                                                                          '7220', utc_time_str,
                                                                          timezone_diff_hrs, get_24hour_data)
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
            weather_data_dict = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
                                                                       elevation_meters, utc_time_str,
                                                                       timezone_diff_hrs, get_24hour_data)

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
        # print(df_extended)

        df_extended = df_extended.drop(columns=['forecast', 'time_utc', 'time_local'])

        # Save the DataFrame to a CSV file
        data_path = config['data_path']
        df_extended.to_csv(f'{data_path}/{csv_name}_Extended.csv', index=False)


def append_to_extended_datafile_main(csv_name, config):
    # Import CSV files
    data_path = config['data_path']
    clean_data_path = config['clean_data_path']
    file = f'{data_path}/{csv_name}_Extended.csv'
    df_extended = pd.read_csv(file)

    clean_data_path = config['clean_data_path']
    file = f'{clean_data_path}/{csv_name}.csv'
    df = pd.read_csv(file)

    startDateTime = df['ts'].iloc[-24]
    endDateTime = ''
    datetime_format = config['datetime_format']

    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # Filter the dataframe to include data within the startDateTime and endDateTime
    if startDateTime and endDateTime:
        # Convert startDateTime and endDateTime to datetime objects
        start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
        end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
        df = df.loc[(df.index >= start_datetime_obj) & (df.index <= end_datetime_obj)]
    elif startDateTime:
        # Convert startDateTime to datetime object
        start_datetime_obj = datetime.strptime(startDateTime, datetime_format)
        df = df.loc[df.index >= start_datetime_obj]
    elif endDateTime:
        # Convert endDateTime to datetime object
        end_datetime_obj = datetime.strptime(endDateTime, datetime_format)
        df = df.loc[df.index <= end_datetime_obj]

    df.reset_index(inplace=True)
    df['ts'] = df['ts'].dt.strftime(datetime_format)

    # last_timestamp = df['ts'].iloc[-1]
    # last_timestamp = datetime.strptime(last_timestamp, datetime_format)

    # # Increment the last timestamp by an hour to get the starting point for the new rows
    # new_start_timestamp = last_timestamp + timedelta(hours=1)

    # # Create a list of new timestamps with incremented hours
    # new_timestamps = [new_start_timestamp + timedelta(hours=i) for i in range(8)]

    # # Create a new DataFrame with the new timestamps
    # new_rows = pd.DataFrame({'ts': new_timestamps})
    # for col in df.columns:
    #     if col != 'ts':
    #         new_rows[col] = df[col].iloc[-1]

    # # Convert the 'ts' column to the desired datetime_format
    # new_rows['ts'] = new_rows['ts'].dt.strftime(datetime_format)

    # # Append the new rows to the original DataFrame
    # df = df.append(new_rows, ignore_index=True)

    tf = TimezoneFinder()

    df_extended_new = pd.DataFrame()
    
    running_chunk = False
    row_counter = 0
    get_elevation = True  # Only get the elevation once per file (to improve speed)
    file_invalid = False

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Convert timestamp to local time
        dt_local = datetime.strptime(row['ts'], config['datetime_format'])
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
            # print(timezone_diff_hrs)
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
            # weather_data_dict_24 = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
            #                                                               elevation_meters, utc_time_str,
            #                                                               timezone_diff_hrs, get_24hour_data)
            weather_data_dict_24 = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
                                                                          '7220', utc_time_str,
                                                                          timezone_diff_hrs, get_24hour_data)
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
            df_extended_new = pd.concat([df_extended_new, row_df_24], axis=0)
        elif not running_chunk:
            # Get the elevation at the given latitude and longitude
            if get_elevation:
                elevation_meters = get_elevation_online(row['latitude'], row['longitude'])
                get_elevation = False

            # Get weather data
            get_24hour_data = False  # Get weather data for the specified local hour
            weather_data_dict = get_weather_data_at_location_and_hour(row['latitude'], row['longitude'],
                                                                       elevation_meters, utc_time_str,
                                                                       timezone_diff_hrs, get_24hour_data)

            # Convert the weather_data_dict to a pandas Series object
            weather_data_series = pd.Series(weather_data_dict)

            # Append the weather_data_series to the original row
            combined_series = pd.concat([row, weather_data_series], axis=0)

            # Append the combined_row to the new DataFrame
            df_extended_new = pd.concat([df_extended_new, combined_series.to_frame().T], axis=0)  # to_frame().T converts the Series to a DataFrame row

    if file_invalid:
        print(f':: ** File {csv_name}_Extended.csv cannot be processed due to invalid entries...')
    else:
        df_extended_new.columns = df_extended_new.columns.str.lower()
        # print(df_extended_new)

        df_extended_new = df_extended_new.drop(columns=['forecast', 'time_utc', 'time_local'])

        # Save the DataFrame to a CSV file
        # print(df_extended_new)
        df_extended = pd.concat([df_extended, df_extended_new], axis=0)
        print(df_extended)
        # df_extended.to_csv(f'{data_path}/{csv_name}_Extended.csv', index=False)
