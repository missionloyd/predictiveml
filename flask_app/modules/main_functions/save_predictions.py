
def save_predictions(config, cli_args):
  from modules.prediction_methods.master_prediction import master_prediction
  from modules.prediction_methods.merge_predictions import merge_predictions
  from modules.prediction_methods.save_predictions import save_predictions

  winners_in_file_path = config['winners_in_file_path']

  results = master_prediction(cli_args, winners_in_file_path, config)
  merged_list = merge_predictions(results, config)
  save_predictions(merged_list, config, cli_args)
            
  return