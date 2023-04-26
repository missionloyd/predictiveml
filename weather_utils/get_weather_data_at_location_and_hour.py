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
    """
    # Parse the datetime string into a datetime object
    dt_local = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')

    # Create a timezone object from the given timezone string
    tz = pytz.timezone(tz_str)

    # Localize the datetime object to the given timezone
    dt_tz = tz.localize(dt_local)

    # Convert the datetime object to UTC
    dt_utc = dt_tz.astimezone(pytz.utc)

    # Format the UTC datetime object as a string
    dt_utc_str = dt_utc.strftime("%Y%m%d%H")

    return dt_utc_str

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

def get_weather_data_at_location_and_hour(latitude, longitude, altitude, query_hour=None):
    """
    Fetches upper air wind speed and direction data from the GFS Model
    for the specified latitude, longitude, altitude, and hour.
    API: https://open-meteo.com/en/docs/gfs-api
    The API dataset is available from past 3 weeks (21 days) to 16 days in the future.

    Parameters:
        latitude (float): The latitude in decimal degrees format.
        longitude (float): The longitude in decimal degrees format.
        altitude (int): The altitude in meters.
        query_hour (str): The forecast hour in YYYYMMDDHH format. Defaults to the current UTC time.

    Returns:
        wind_speed (float): The wind speed in meters per second.
        wind_direction (float): The wind direction in degrees.
    """
    # Set the query hour to the current UTC hour if not specified
    if query_hour is None:
        query_hour_dt = datetime.utcnow().strftime('%Y%m%d%H')
    else:
        # Parse the forecast time string into a datetime object
        query_hour_dt = datetime.strptime(query_hour, '%Y%m%d%H')

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
            query_hour = query_hour_dt.strftime('%Y%m%d%H')
            print('Warning: Forecast time limited to 16 days from the current UTC time.')

    # Calculate the pressure at the specified altitude
    pressure_hpa = pressure_at_altitude(altitude)/100
    # Table from https://open-meteo.com/en/docs/gfs-api
    lookup_pressure = np.array([1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,70,50,40,30,20,15,10])

    # Find the nearest pressure level
    nearest_idx = np.abs(lookup_pressure - pressure_hpa).argmin()
    nearest_pressure = lookup_pressure[nearest_idx]

    query_windspeed_air_string = f'windspeed_{nearest_pressure}hPa'
    query_winddirection_air_string = f'winddirection_{nearest_pressure}hPa'

    query_windspeed_ground_string = 'windspeed_10m'
    query_temperature_ground_string = 'temperature_2m'
    query_relativehumidity_ground_string = 'relativehumidity_2m'

    hourly_params = [query_temperature_ground_string, 
                     query_relativehumidity_ground_string, 
                     query_windspeed_ground_string,
                     query_windspeed_air_string,
                     query_winddirection_air_string]

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
        'current_weather': 'true'
    }

    # Determine if the forecast time is in the future or past
    if isForecast:
        url = url_template_forecast.format(params=urlencode(params_forecast, doseq=True))
    else:
        # Switch to access historical data
        url_template_past = 'https://archive-api.open-meteo.com/v1/archive?{params}'

        # Find the nearest altitude (only 100m and 10m are available from GFS)
        lookup_past_altitude = np.array([100, 10])
        nearest_idx = np.abs(lookup_past_altitude - altitude).argmin()
        nearest_altitude = lookup_past_altitude[nearest_idx]

        query_windspeed_string = f'windspeed_{nearest_altitude}m'
        query_winddirection_string = f'winddirection_{nearest_altitude}m'

        hourly_params = [query_temperature_ground_string, query_windspeed_string, query_winddirection_string]

        params_past = {
            'latitude': str(latitude),
            'longitude': str(longitude),
            'elevation': str(altitude),
            'start_date': query_hour_dt.strftime('%Y-%m-%d'),
            'end_date': query_hour_dt.strftime('%Y-%m-%d'),
            'hourly': hourly_params,
            'current_weather': 'true'}
        
        url = url_template_past.format(params=urlencode(params_past, doseq=True))

    # print(url)
    print(f':: --- Query weather data @ GPS[{latitude, longitude})-({altitude})] @ open-meteo.com.')

    # Send a request to the API endpoint
    response = requests.get(url)

    # Debug check response
    #print(response.status_code)
    #print(json.dumps(json.loads(response.content), indent=2))

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
        wind_speed_ground = np.nan
        wind_speed_air = np.nan
        wind_dir_air = np.nan

        if len(time_idx) > 0:
            # Extract the wind speed and direction data for the matching time
            temp = data['hourly'][query_temperature_ground_string][time_idx[0]]

            if isForecast:
                rel_humidity = data['hourly'][query_relativehumidity_ground_string][time_idx[0]]
                wind_speed_ground = data['hourly'][query_windspeed_ground_string][time_idx[0]]
                wind_speed_air = data['hourly'][query_windspeed_air_string][time_idx[0]]
                wind_dir_air = data['hourly'][query_winddirection_air_string][time_idx[0]]
            else:
                wind_speed_air = data['hourly']['windspeed_100m'][time_idx[0]]
                wind_dir_air = data['hourly']['winddirection_100m'][time_idx[0]]

        # Convert wind direction from 0 to 360 to -180 to 180
        wind_dir_air = (wind_dir_air + 180) % 360 - 180

        # Return the weather data in a dictionary
        weather_dict = {'temp': temp, 
                        'rel_humidity': rel_humidity, 
                        'wind_speed_ground': wind_speed_ground, 
                        'wind_speed_air': wind_speed_air, 
                        'wind_dir_air': wind_dir_air}
        
        return weather_dict
    else:
        # Raise an error if the request was not successful
        response.raise_for_status()
