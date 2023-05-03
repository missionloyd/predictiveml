# Description: This file contains functions for fetching weather data from the GFS Model
#              for the specified latitude, longitude, altitude, and hour.
#              API: https://open-meteo.com/en/docs/gfs-api
#              The API dataset is available from past 3 weeks (21 days) to 16 days in the future.
#              In addition, we query historical datasets for older dates.

# ==================================================================================
import pytz

def local_time_to_utc_str(dt_str, tz_str):
    """
    Converts a datetime string in the specified timezone to UTC.

    Args:
        dt_str (str): A datetime string in the format 'YYYY-MM-DD HH:MM'.
        tz_str (str): A string representing the timezone.

    Returns:
        str: A datetime string in UTC in the format 'YYYYMMDDHH'.
        int: The time difference in hours UTC - local time.
    """
    # Parse the datetime string into a datetime object
    dt_local = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')

    # Create a timezone object from the given timezone string
    tz = pytz.timezone(tz_str)

    # Localize the datetime object to the given timezone
    dt_tz = tz.localize(dt_local)

    # Calculate the time difference from the current UTC time
    time_diff_hrs = int( dt_tz.utcoffset().total_seconds() / 3600 )

    # Convert the datetime object to UTC
    dt_utc = dt_tz.astimezone(pytz.utc)
    
    # Format the UTC datetime object as a string
    dt_utc_str = dt_utc.strftime("%Y-%m-%dT%H:%M")

    return dt_utc_str, time_diff_hrs

# ==================================================================================

import math

def pressure_at_altitude(h):
    """
    Calculates the atmospheric pressure at a given altitude, taking into account the decreasing temperature with altitude.

    Args:
        h (float): Altitude in meters.

    Returns:
        float: Atmospheric pressure in Pa.
    """
    p0 = 101325  # sea level standard atmospheric pressure in Pa
    g = 9.80665  # acceleration due to gravity in m/s^2
    M = 0.0289644  # molar mass of dry air in kg/mol
    R = 8.31447  # ideal gas constant in J/(mol*K)
    L = 0.0065  # temperature lapse rate in K/m
    T0 = 288.15  # standard temperature at sea level in K
    g_over_R_L = g / (R * L)

    # Calculate the temperature at the given altitude
    T = T0 + L * h

    # Calculate the pressure at the given altitude using the full barometric formula
    pressure = p0 * math.pow((T0 / T), g_over_R_L * M)

    return pressure

# ==================================================================================

import requests
from json.decoder import JSONDecodeError

def get_elevation_online(lat, lng):
    """
    Query the USGS 3D Elevation Program (3DEP) Elevation Query web service for elevation data.
    
    Args:
        lat (float): Latitude of the location in decimal degrees (WGS 84).
        lng (float): Longitude of the location in decimal degrees (WGS 84).

    Returns:
        float: Elevation at the given location in meters, or None if there was an error.
    """
    # Construct the URL for the 3DEP Elevation Query web service using the given latitude and longitude
    url = f'https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples?returnGeometry=false&geometryType=esriGeometryPoint&geometry={{"x":{lng},"y":{lat},"spatialReference":{{"wkid":4326}}}}&f=json'
    
    # Send an HTTP GET request to the web service
    response = requests.get(url)

    # Try to parse the JSON response
    try:
        data = response.json()
    except JSONDecodeError:
        print(f"Error: Unable to parse JSON response. Server returned: {response.text}")
        return None

    # Check if elevation data is present in the response
    if 'samples' in data and data['samples']:
        elevation_meters = float(data['samples'][0]['value'])
    else:
        print("Error: Elevation data not found in the response.")
        return None

    return elevation_meters

# ==================================================================================

import requests
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urlencode
import json
import time

def get_weather_data_at_location_and_hour(latitude, longitude, altitude, query_hour_str, timezone_diff_hrs, get_24hour_data=False):
    """
    Fetches upper air wind speed and direction data from the GFS Model
    for the specified latitude, longitude, altitude, and hour.
    API: https://open-meteo.com/en/docs/gfs-api
    The API dataset is available from past 3 weeks (21 days) to 16 days in the future.

    Parameters:
        latitude (float): The latitude in decimal degrees format.
        longitude (float): The longitude in decimal degrees format.
        altitude (int): The altitude in meters.
        query_hour_str (str): The forecast hour in '%Y-%m-%dT%H:%M' format. Defaults to the current UTC time.

    Returns:
        wind_speed (float): The wind speed in meters per second.
        wind_direction (float): The wind direction in degrees.
    """
    # Set the query hour to the current UTC hour if not specified
    if query_hour_str is None:
        query_hour_dt = datetime.utcnow().strftime('%Y-%m-%dT%H:%M')
    else:
        # Parse the forecast time string into a datetime object
        query_hour_dt = datetime.strptime(query_hour_str, '%Y-%m-%dT%H:%M')

        # Calculate the time difference from the current UTC time
        time_diff = query_hour_dt - datetime.utcnow()

        # Check if the forecast time is in the future
        if time_diff.days > -21:
            isForecast = True
        else:
            isForecast = False

        # Check if the forecast time is more than 16 days in the future
        if time_diff.days > 16:
            # Limit the forecast time to 16 days in the future
            query_hour_dt = datetime.utcnow() + timedelta(days=16)
            query_hour = query_hour_dt.strftime('%Y-%m-%dT%H:%M')
            print('Warning: Forecast time limited to 16 days from the current UTC time.')

    # Surface level parameters
    query_temperature_ground_string = 'temperature_2m'
    query_relativehumidity_ground_string = 'relativehumidity_2m'
    query_surface_pressure_string = 'surface_pressure'
    query_cloudcover_string = 'cloudcover'
    query_direct_radiation_string = 'direct_radiation'
    quert_precipitation_string = 'precipitation'
    query_windspeed_ground_string = 'windspeed_10m'
    query_winddirection_ground_string = 'winddirection_10m'

    hourly_params = [query_temperature_ground_string, 
                     query_relativehumidity_ground_string,
                     query_surface_pressure_string,
                     query_cloudcover_string,
                     query_direct_radiation_string,
                     quert_precipitation_string,
                     query_windspeed_ground_string,
                     query_winddirection_ground_string]

    # Define the API endpoint URL
    url_template_forecast = 'https://api.open-meteo.com/v1/gfs?{params}'

    # Define the request parameters
    params_forecast = {
        'latitude': str(latitude),
        'longitude': str(longitude),
        'elevation': str(altitude),
        'forecast_days': 16,
        'start_date': query_hour_dt.strftime('%Y-%m-%d'),
        'end_date': query_hour_dt.strftime('%Y-%m-%d'),
        'hourly': hourly_params,
        'current_weather': 'false'
    }

    # Determine if the forecast time is in the future or past
    if isForecast:
        url = url_template_forecast.format(params=urlencode(params_forecast, doseq=True))
    else:
        # Switch to access historical data
        url_template_past = 'https://archive-api.open-meteo.com/v1/archive?{params}'

        params_past = {
            'latitude': str(latitude),
            'longitude': str(longitude),
            'elevation': str(altitude),
            'start_date': query_hour_dt.strftime('%Y-%m-%d'),
            'end_date': query_hour_dt.strftime('%Y-%m-%d'),
            'hourly': hourly_params,
            'current_weather': 'false'}
        
        url = url_template_past.format(params=urlencode(params_past, doseq=True))

    # print(url)
    #print(f':: --- Query weather @ GPS[{latitude, longitude})-({altitude})] @ open-meteo.com.')

    # Send a request to the API endpoint, using Try/Except to catch errors

    total_timeout_seconds = 600  # 10 minutes
    request_timeout = 30
    retry_interval_seconds = 30
    end_time = time.time() + total_timeout_seconds

    while time.time() < end_time:
        try:
            response = requests.get(url, timeout=request_timeout)
            break  # If the request is successful, exit the loop
        except requests.exceptions.Timeout:
            print(f':: -- Timeout occurred, retrying in {retry_interval_seconds} seconds...')
            time.sleep(retry_interval_seconds)
            # wait longer if requests.exceptions.ConnectionError
        except requests.exceptions.ConnectionError:
            print(f':: -- Connection rejected by API, retrying in {retry_interval_seconds*10} seconds...')
            time.sleep(retry_interval_seconds * 10)
    else:
        raise Exception(f':: -- Failed to get a response within {total_timeout_seconds}.')
    
    # Debug check response
    #print(response.status_code)
    #print(json.dumps(json.loads(response.content), indent=2))

    timedelta_hours = timedelta(hours = timezone_diff_hrs)

    if response.status_code == 200:
        # Extract the wind speed and direction data from the response JSON
        data = response.json()
    
        # Round the forecast time to the nearest hour
        if query_hour_dt.minute >= 30:
            query_hour_dt_rounded = query_hour_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            query_hour_dt_rounded = query_hour_dt.replace(minute=0, second=0, microsecond=0)

        # Find the index of the time that matches the forecast hour
        query_hour_str = query_hour_dt_rounded.strftime('%Y-%m-%dT%H:%M')
        time_idx = np.where(np.array(data['hourly']['time']) == query_hour_str)[0]

        temp = np.nan
        rel_humidity = np.nan
        pressure = np.nan
        cloud_cover = np.nan
        direct_radiation = np.nan
        precipitation = np.nan
        wind_speed_ground = np.nan
        wind_dir_ground = np.nan

        if len(time_idx) > 0:
            # Extract the wind speed and direction data for the matching time
            if get_24hour_data:
                time_utc = data['hourly']['time']
                time_local = [(datetime.strptime(t, '%Y-%m-%dT%H:%M') + timedelta_hours).strftime('%Y-%m-%dT%H:%M') for t in time_utc]
                temp = data['hourly'][query_temperature_ground_string]
                rel_humidity = data['hourly'][query_relativehumidity_ground_string]
                pressure = data['hourly'][query_surface_pressure_string]
                cloud_cover = data['hourly'][query_cloudcover_string]
                direct_radiation = data['hourly'][query_direct_radiation_string]
                precipitation = data['hourly'][quert_precipitation_string]
                wind_speed_ground = data['hourly'][query_windspeed_ground_string]
                wind_dir_ground = data['hourly'][query_winddirection_ground_string]
            else:
                time_utc = data['hourly']['time'][time_idx[0]]
                time_local = (datetime.strptime(time_utc, '%Y-%m-%dT%H:%M') + timedelta_hours).strftime('%Y-%m-%dT%H:%M')
                temp = data['hourly'][query_temperature_ground_string][time_idx[0]]
                rel_humidity = data['hourly'][query_relativehumidity_ground_string][time_idx[0]]
                pressure = data['hourly'][query_surface_pressure_string][time_idx[0]]
                cloud_cover = data['hourly'][query_cloudcover_string][time_idx[0]]
                direct_radiation = data['hourly'][query_direct_radiation_string][time_idx[0]]
                precipitation = data['hourly'][quert_precipitation_string][time_idx[0]]
                wind_speed_ground = data['hourly'][query_windspeed_ground_string][time_idx[0]]
                wind_dir_ground = data['hourly'][query_winddirection_ground_string][time_idx[0]]

        weather_dict = {'forecast': isForecast,
                        'time_UTC': time_utc,
                        'time_local': time_local,
                        'temp_C': temp, 
                        'rel_humidity_%': rel_humidity,
                        'surface_pressure_hPa': pressure,
                        'cloud_cover_%': cloud_cover,
                        'direct_radiation_W/m2': direct_radiation,
                        'precipitation_mm': precipitation,
                        'wind_speed_ground_Km/h': wind_speed_ground,
                        'wind_dir_ground_deg': wind_dir_ground}
        
        return weather_dict
    else:
        # Raise an error if the request was not successful
        response.raise_for_status()
