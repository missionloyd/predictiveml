def insert(config, cli_args):
  from modules.prediction_methods.master_prediction import master_prediction
  from modules.prediction_methods.merge_predictions import merge_predictions
  from modules.db_management.insert_data import insert_data

  winners_in_file_path = config['winners_in_file_path']

  results = master_prediction(cli_args, winners_in_file_path, config)
  merged_list = merge_predictions(results, config)
  insert_data(merged_list, config)

  return