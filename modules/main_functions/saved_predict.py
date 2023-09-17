def saved_predict(config, cli_args):
  from modules.prediction_methods.saved_on_demand_prediction import saved_on_demand_prediction
  from modules.logging_methods.main import logger, api_logger

  required_columns = ['time_step', 'datelevel']

  if not all(key in cli_args for key in required_columns):
    logger(f"Required arguments missing for prediction. Please provide {required_columns}.")
    return
  
  results = saved_on_demand_prediction(cli_args, config)
  # logger(results)
  api_logger(results, config)
  
  return