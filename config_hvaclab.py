from modules.utils.get_file_names import get_file_names
from modules.utils.get_add_features import get_add_features

def load_config():
  path = '.'
  data_path = f'{path}/hvaclab_data'
  tmp_path = f'{path}/models/tmp'
  log_path = f'{path}/logs'
  results_file_path = f'{path}/results.csv'
  results_header = ['model_type', 'bldgname', 'y_column', 'imputation_method', 'feature_method', 'n_feature', 'updated_n_feature', 'time_step', 'datelevel', 'rmse', 'mae', 'mape', 'model_file', 'model_data_path', 'building_file', 'selected_features_delimited']
  y_column = ['total_ele (kw)']
  exclude_column = ['ts', 'bldgname']
  exclude_file = ['']
  file_path = f'{data_path}/data.csv'
  file_list = get_file_names(data_path, exclude_file)
  add_feature = get_add_features(file_path, y_column + exclude_column)
  header = ['ts'] + y_column + add_feature
  n_feature = list(range(0, len(add_feature)))

  config = {
    # settings
    'job_id': 0,
    'table': '',
    'path': path,
    'data_path': data_path,
    'tmp_path': tmp_path,
    'log_path': log_path,
    'results_file_path': results_file_path,
    'building_file': file_list,
    'exclude_file': exclude_file,
    'exclude_column': exclude_column,
    'update_add_feature': False,
    'save_preprocessed_file': True,
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
    'y_column_mapping': {
      'total_ele (kw)': 'electricity',
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
    'model_type': ["xgboost"],
    'imputation_method': ['linear_interpolation'],
    'feature_method': ['rfecv', 'lassocv'],
    'datelevel': ['hour'],
    'time_step': [1],             # window size of the sliding window technique and unit length of forecasts
    'train_test_split': 0.8,
    'train_ratio_threshold': 0.8, # minimum percent non-nans in training set
    'test_ratio_threshold': 0.8,  # minimum percent non-nans in testing set
    'datetime_format': '%m/%d/%Y %H:%M',
    'startDateTime': '',
    'endDateTime': '',
        
    # hyperparameters
    'n_feature': n_feature,
    'n_fold': 5,                
    'minutes_per_model': 2,
    'temperature': -1,
    'target_error': 'mape'
  }

  return config
