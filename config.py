import json, sys, warnings, os
from modules.utils.get_file_names import get_file_names
from modules.logging_methods.main import setup_logger

def load_config(job_id):
  setup_logger(job_id or 0)

  path = '.'
  data_path = f"{path}/clean_data_extended"
  tmp_path = f"{path}/models/tmp"
  file_list = get_file_names(data_path)
  results_header = ['model_type', 'bldgname', 'y_column', 'imputation_method', 'feature_method', 'n_feature', 'updated_n_feature', 'time_step', 'datelevel', 'rmse', 'mae', 'mape', 'model_file', 'model_data_path', 'building_file']
  y_column = ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr', 'present_co2_tons']
  add_feature = ['temp_c', 'rel_humidity_%', 'surface_pressure_hpa', 'cloud_cover_%', 'direct_radiation_w/m2', 'precipitation_mm', 'wind_speed_ground_km/h', 'wind_dir_ground_deg']
  header = ['ts'] + y_column + add_feature
  n_feature = list(range(1, len(add_feature)))

  config = {
    # settings
    "path": path,
    "data_path": data_path,
    "tmp_path": tmp_path,
    "building_file": file_list,
    "exclude_file": ["Summary_Report_Extended.csv"],
    "exclude_column": ["present_co2_tons"],
    "update_add_feature": False,
    "save_preprocessed_file": False,
    "save_model_file": False,
    "save_model_plot": False,
    "save_at_each_delta": True,
    "min_number_of_days": 365,      # unused at the moment
    "n_jobs": -1,
    "batch_size": 8,
    "memory_limit": 102400,
    "updated_n_feature": n_feature, # placeholder
    "y_column": y_column,
    "add_feature": add_feature,
    "header": header,
    "results_header": results_header,
    "y_column_mapping": {
        'present_elec_kwh': 'electricity',
        'present_htwt_mmbtu': 'hot_water',
        'present_wtr_usgal': 'water',
        'present_chll_tonhr': 'chilled_water',
        'present_co2_tons': 'co2_emissions'
    },
    
    # training_scope
    "model_type": ["xgboost", "solos", "ensembles"],
    "imputation_method": ["linear_regression","linear_interpolation", "prophet", "lstm"],
    "feature_method": ["rfecv", "lassocv"],
    
    # hyperparameters
    "n_feature": n_feature,
    "n_fold": 5,                  # feature_method n_fold     
    "minutes_per_model": 2,
    "split_rate": 0.7,            # minimum percent non-nans and split index
    "time_step": [1],
    "datelevel": ["year"],
  }

  return config
