import json, sys, warnings, os
from modules.utils.get_file_names import get_file_names

def load_config():

  path = '.'
  print()

  data_path = f"{path}/clean_data_extended"
  tmp_path = f"{path}/models/tmp"
  file_list = get_file_names(data_path)
  results_header = ['model_type', 'bldgname', 'y_column', 'imputation_method', 'feature_method', 'n_feature', 'time_step', 'rmse', 'mae', 'mape', 'model_file']
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
    "exclude_column": ["present_co2_tons"],
    "preprocess_files": False,
    "save_model_file": False,
    "save_model_plot": False,
    "min_number_of_days": 365,
    "batch_size": 32,
    "memory_limit": 102400,
    "y_column": y_column,
    "add_feature": add_feature,
    "header": header,
    "results_header": results_header,

    # training_scope
    "model_type": ["ensembles", "solos"],
    "imputation_method": ["linear_regression", "linear_interpolation", "prophet", "lstm"],
    "feature_method": ["rfecv", "lassocv"],
    
    # hyperparameters
    "n_feature": n_feature,
    "n_fold": 5,
    "time_step": [1, 8, 12, 24],
    "minutes_per_model": 2,
    "split_rate": 0.8
  }

  return config
