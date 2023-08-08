from db_management import create_table

key_name = 'key'

science_schema = f"""
  ts TIMESTAMP WITHOUT TIME ZONE, 
  bldgname VARCHAR, 
  present_elec_kwh NUMERIC,
  present_htwt_mmbtuh NUMERIC,
  present_chll_tonh NUMERIC,
  present_co2_tonh NUMERIC,
  latitude NUMERIC,
  longitude NUMERIC,
  year INT,
  month INT,
  day INT,
  hour INT,
  CONSTRAINT {key_name} UNIQUE (bldgname, ts)
"""

spaces_schema = f"""
  ts TIMESTAMP WITHOUT TIME ZONE, 
  bldgname VARCHAR, 
  present_elec_kwh NUMERIC,
  present_htwt_mmbtuh NUMERIC,
  present_chll_tonh NUMERIC,
  present_co2_tonh NUMERIC,
  historical_elec_kwh NUMERIC,
  historical_htwt_mmbtuh NUMERIC,
  historical_chll_tonh NUMERIC,
  historical_co2_tonh NUMERIC,
  predicted_elec_kwh NUMERIC,
  predicted_htwt_mmbtuh NUMERIC,
  predicted_chll_tonh NUMERIC,
  predicted_co2_tonh NUMERIC,
  latitude NUMERIC,
  longitude NUMERIC,
  year INT,
  month INT,
  day INT,
  hour INT,
  CONSTRAINT {key_name} UNIQUE (bldgname, ts)
"""

create_table(table="spaces", schema=spaces_schema)
