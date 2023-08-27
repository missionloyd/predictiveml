from modules.utils.get_file_names import get_file_names
from modules.utils.get_add_features import get_add_features
from modules.utils.create_results_file_path import create_results_file_path

def load_config(path='', data_path='', clean_data_path=''):
  path = path or '.'
  data_path = data_path or f'{path}/building_data'
  clean_data_path = clean_data_path or f'{path}/clean_data'
  prediction_data_path = f'{path}/prediction_data'
  tmp_path = f'{path}/models/tmp'
  imp_path = f'{path}/models/imp'
  log_path = f'{path}/logs'
  results_file = 'results.csv'
  results_file_path = create_results_file_path(path, results_file)
  args_file_path = f'{path}/models/tmp/_args'
  winners_in_file_path = f'{path}/models/tmp'
  winners_out_file_path = f'{path}/models/tmp'
  winners_out_file = f'{winners_out_file_path}/_winners.out'
  results_header = ['model_type', 'bldgname', 'y_column', 'imputation_method', 'feature_method', 'n_feature', 'updated_n_feature', 'time_step', 'datelevel', 'rmse', 'mae', 'mape', 'model_file', 'model_data_path', 'building_file', 'selected_features_delimited']
  y_column = ['present_elec_kwh', 'present_htwt_mmbtuh', 'present_wtr_usgal', 'present_chll_tonh']
  exclude_column = ['ts', 'bldgname', 'present_co2_tonh', 'campus', 'historical_elec_kwh', 'historical_htwt_mmbtuh', 'historical_wtr_usgal', 'historical_chll_tonh', 'historical_co2_tonh', 'latitude', 'longitude', 'year', 'month', 'day', 'hour']
  exclude_file = ['Summary_Report_Extended.csv', 'East_Campus_Data_Extended.csv', 'West_Campus_Data_Extended.csv']
  file_list = get_file_names(data_path, exclude_file)
  add_feature = get_add_features(file_list, data_path, y_column + exclude_column)
  header = ['ts'] + y_column + add_feature
  n_feature = list(range(0, len(add_feature)))

  config = {
    # preprocessing/training scope
    'model_type': ["xgboost"],                      # fastest and most lightweight setting
    'imputation_method': ['zero_fill'],  # fastest and most lightweight setting
    # 'model_type': ["xgboost", "solos", "ensembles"],
    # 'imputation_method': ['zero_fill', 'linear_interpolation', 'linear_regression', 'prophet', 'lstm'],
    'feature_method': ['rfecv', 'lassocv'],
    'time_step': [24],            # window size of the sliding window technique and unit length of forecasts
    'datelevel': ['month'],
    'train_test_split': 0.5,
    'train_ratio_threshold': 0.5, # minimum percent non-nans in training set
    'test_ratio_threshold': 0.5,  # minimum percent non-nans in testing set
    'datetime_format': '%Y-%m-%dT%H:%M:%S',
    'startDateTime': '',
    'endDateTime': '',
        
    # hyperparameters
    'n_feature': n_feature,
    'n_fold': 5,                
    'minutes_per_model': 2,
    'temperature': -1,
    'target_error': 'mape',

    # settings
    'job_id': 0,
    'table': 'spaces',
    'path': path,
    'data_path': data_path,
    'clean_data_path': clean_data_path,
    'prediction_data_path': prediction_data_path,
    'tmp_path': tmp_path,
    'imp_path': imp_path,
    'log_path': log_path,
    'results_file': results_file,
    'results_file_path': results_file_path,
    'building_file': file_list,
    'exclude_file': exclude_file,
    'exclude_column': exclude_column,
    'update_add_feature': False,
    'save_preprocessed_files': False,
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
    'args_file_path': args_file_path,
    'winners_in_file_path': winners_in_file_path,
    'winners_out_file_path': winners_out_file_path,
    'winners_out_file': winners_out_file,
    "y_column_mapping": {
      'present_elec_kwh': 'electricity',
      'present_htwt_mmbtuh': 'hot_water',
      'present_wtr_usgal': 'water',
      'present_chll_tonh': 'chilled_water',
      'present_co2_tonh': 'co2_emissions'
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
      # 'models/imp',
      # 'prediction_data'
    ],
    'CO2EmissionFactor': 0.00069,
  }

  return config