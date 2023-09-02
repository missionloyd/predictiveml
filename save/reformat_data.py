import csv
import numpy as np 
import pandas as pd

file_list = ['Anthropology_Data.csv', 'EERB_Data.csv', 'Enzi-STEM_Data.csv', 'Science_Initiative_Data.csv']

for f in file_list:
  df = pd.read_csv(f)
  df['ts'] = df['ts'].str.replace(' ', 'T')
  df = df.drop(columns=['predicted_elec_kwh',  'predicted_htwt_mmbtuh',  'predicted_chll_tonh',  'predicted_co2_tonh'])
  historical_cols = ['historical_elec_kwh',  'historical_htwt_mmbtuh',  'historical_chll_tonh',  'historical_co2_tonh']

  for index, row in df.iterrows():
    df.loc[index, historical_cols] = np.nan

  if f == 'Anthropology_Data.csv':
    df['campus'] = 'West'
  elif f == 'EERB_Data.csv':
    df['campus'] = 'West'
  elif f == 'Enzi-STEM_Data.csv':
    df['campus'] = 'West'
  elif f == 'Science_Initiative_Data.csv':
    df['campus'] = 'West'
   
  df = df.reindex(columns=['campus', 'bldgname', 'ts', 'present_elec_kwh', 'present_htwt_mmbtuh', 'present_wtr_usgal', 'present_chll_tonh', 'present_co2_tonh'] +  [f'{col}' for col in historical_cols] + ['latitude', 'longitude', 'year', 'month', 'day', 'hour'])
  
  print(df)

  df.to_csv(f, index=None)