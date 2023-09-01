import csv
import pandas as pd

file_list = ['Anthropology.csv', 'EERB.csv', 'Enzi-STEM.csv', 'Science_Initiative.csv']

for f in file_list:
  df = pd.read_csv(f)
  df['ts'] = df['ts'].str.replace(' ', 'T')
  df = df.drop(columns=['predicted_elec_kwh',  'predicted_htwt_mmbtuh',  'predicted_chll_tonh',  'predicted_co2_tonh'])

  if f == 'Anthropology.csv':
    df['campus'] = 'West'
  elif f == 'EERB.csv':
    df['campus'] = 'West'
  elif f == 'Enzi-STEM.csv':
    df['campus'] = 'West'
  elif f == 'Science_Initiative.csv':
    df['campus'] = 'West'
   
  df = df.reindex(columns=['campus', 'bldgname', 'ts', 'present_elec_kwh', 'present_htwt_mmbtuh', 'present_wtr_usgal', 'present_chll_tonh', 'present_co2_tonh', 'latitude', 'longitude', 'year', 'month', 'day', 'hour']) 
  
  print(df)

  df.to_csv(f, index=None)