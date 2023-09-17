import csv
import numpy as np 
import pandas as pd

file_list = ['Anthropology_Data_Extended.csv', 'EERB_Data_Extended.csv', 'Enzi-STEM_Data_Extended.csv', 'Science_Initiative_Data_Extended.csv']

for f in file_list:
  df = pd.read_csv(f)
  df['ts'] = df['ts'].str.replace(' ', 'T')
  # df = df.drop(columns=['predicted_elec_kwh',  'predicted_htwt_mmbtuh',  'predicted_chll_tonh',  'predicted_co2_tonh'])
  historical_cols = ['historical_elec_kwh',  'historical_htwt_mmbtuh',  'historical_chll_tonh',  'historical_wtr_usgal', 'historical_co2_tonh']

  for index, row in df.iterrows():
    df.loc[index, historical_cols] = np.nan

  df['campus'] = 'West'
  df['historical_wtr_usgal'] = np.nan
   
  df = df.reindex(columns=['campus', 'bldgname', 'ts', 'present_elec_kwh', 'present_htwt_mmbtuh', 'present_wtr_usgal', 'present_chll_tonh', 'present_co2_tonh'] +  [f'{col}' for col in historical_cols] + ['latitude', 'longitude', 'year', 'month', 'day', 'hour'] + ['temp_c','rel_humidity_%','surface_pressure_hpa','cloud_cover_%','direct_radiation_w/m2','precipitation_mm','wind_speed_ground_km/h','wind_dir_ground_deg'])
  print(df)

  df.to_csv(f, index=None)