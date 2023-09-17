def predict(config, cli_args):
  from modules.prediction_methods.on_demand_prediction import on_demand_prediction
  from modules.logging_methods.main import logger, api_logger

  winners_in_file_path = config['winners_in_file_path']
  required_columns = ['building_file', 'y_column', 'time_step', 'datelevel']

  if not all(key in cli_args for key in required_columns):
    logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
    return
  
  # Call the make_prediction function with the provided arguments
  results = on_demand_prediction(cli_args, winners_in_file_path, config)

  # save prediction(s) for info logs and expose to api
  logger(results)
  api_logger(results, config)
  return