import json, sys, warnings, os
from modules.utils.get_file_names import get_file_names
from modules.logging_methods.main import setup_logger
from modules.utils.get_add_features import get_add_features

def load_config(job_id):
  setup_logger(job_id or 0)

  path = '.'
  data_path = f'{path}/building_data'
  tmp_path = f'{path}/models/tmp'
  results_file_path = f'{path}/results.csv'
  results_header = ['model_type', 'bldgname', 'y_column', 'imputation_method', 'feature_method', 'n_feature', 'updated_n_feature', 'time_step', 'datelevel', 'rmse', 'mae', 'mape', 'model_file', 'model_data_path', 'building_file', 'selected_features_delimited']
  y_column = ['present_elec_kwh', 'present_htwt_mmbtu', 'present_wtr_usgal', 'present_chll_tonhr']
  exclude_column = ['ts', 'bldgname', 'present_co2_tons', 'campus', 'historical_elec_kwh', 'historical_htwt_mmbtu', 'historical_wtr_usgal', 'historical_chll_tonhr', 'historical_co2_tons', 'latitude', 'longitude', 'year', 'month', 'day', 'hour']
  exclude_file = ['Summary_Report_Extended.csv']
  file_path = f'{data_path}/Summary_Report_Extended.csv'
  file_list = get_file_names(data_path, exclude_file)
  add_feature = get_add_features(file_path, y_column + exclude_column)
  header = ['ts'] + y_column + add_feature
  n_feature = list(range(1, len(add_feature)))  # start from 1 because sliding window technique will count as a feature

  config = {
    # settings
    'path': path,
    'data_path': data_path,
    'tmp_path': tmp_path,
    'results_file_path': results_file_path,
    'building_file': file_list,
    'exclude_file': exclude_file,
    'exclude_column': exclude_column,
    'update_add_feature': False,
    'save_preprocessed_file': False,
    'save_model_file': False,
    'save_model_plot': False,
    'save_at_each_delta': True,
    'n_jobs': -1,
    'batch_size': 8,
    'memory_limit': 102400,
    'updated_n_feature': -1,
    'y_column': y_column,
    'add_feature': add_feature,
    'header': header,
    'results_header': results_header,
    'selected_features_delimited': '',
    "y_column_mapping": {
      'present_elec_kwh': 'electricity',
      'present_htwt_mmbtu': 'hot_water',
      'present_wtr_usgal': 'water',
      'present_chll_tonhr': 'chilled_water',
      'present_co2_tons': 'co2_emissions'
    },
    'directories_to_prune': [
      'logs/api_log',
      'logs/debug_log',
      'logs/error_log',
      'logs/flask_log',
      'logs/info_log',
      'models/tmp',
      'models/ensembles',
      'models/solos',
      'models/xgboost',
    ],

    # preprocessing/training scope
    'model_type': ["xgboost", "solos", "ensembles"],
    'imputation_method': ['linear_interpolation', 'linear_regression', 'prophet', 'lstm'],
    'feature_method': ['rfecv', 'lassocv'],
    'datelevel': ['hour'],
    'time_step': [24],            # window size of the sliding window technique and unit length of forecasts
    'train_test_split': 0.7,
    'train_ratio_threshold': 0.7, # minimum percent non-nans in training set
    'test_ratio_threshold': 0.7,  # minimum percent non-nans in testing set
    'datetime_format': '%Y-%m-%dT%H:%M:%S',
    'startDateTime': '',
    'endDateTime': '2023-02-04T00:00:00',
        
    # hyperparameters
    'n_feature': n_feature,
    'n_fold': 5,                
    'minutes_per_model': 2,
    'temperature': -1,
    'target_error': 'mape'
  }

  return config
