import json
from itertools import product    

# Generate a list of arguments for model training
def create_args(config, cli_args):
  arguments = []

  if cli_args['building_file']:
    config['building_file'] = [cli_args['building_file']]

  if cli_args['time_step'] and cli_args['datelevel']:
    config['time_step'] = [cli_args['time_step']]
    config['datelevel'] = [cli_args['datelevel']]

  for combo in product(
    config["building_file"],
    config["y_column"],
    config["imputation_method"],
    config["model_type"],
    config["feature_method"],
    config["n_feature"],
    config["time_step"],
    config["datelevel"],
    [config["header"]],
    [config["data_path"]],
    [config["add_feature"]],
    [config["min_number_of_days"]],
    [config["exclude_column"]],
    [config["n_fold"]],
    [config["split_rate"]],
    [config["minutes_per_model"]],
    [config["memory_limit"]],
    [config["save_model_file"]],
    [config["save_model_plot"]],
    [config["path"]],
    [config["save_preprocessed_file"]],
    [config["updated_n_feature"]],
  ):
    argument_dict = {
      "building_file": combo[0],
      "y_column": combo[1],
      "imputation_method": combo[2],
      "model_type": combo[3],
      "feature_method": combo[4],
      "n_feature": combo[5],
      "time_step": combo[6],
      "datelevel": combo[7],
      "header": combo[8],
      "data_path": combo[9],
      "add_feature": combo[10],
      "min_number_of_days": combo[11],
      "exclude_column": combo[12],
      "n_fold": combo[13],
      "split_rate": combo[14],
      "minutes_per_model": combo[15],
      "memory_limit": combo[16],
      "save_model_file": combo[17],
      "save_model_plot": combo[18],
      "path": combo[19],
      "save_preprocessed_file": combo[20],
      "updated_n_feature": combo[21],
    }
    arguments.append(argument_dict)

  # Generate a list of arguments for preprocessing
  preprocessing_arguments = []
  for combo in product(
    config["building_file"],
    config["y_column"],
    config["imputation_method"],
    [config["header"]],
    [config["data_path"]],
    [config["tmp_path"]],
    [config["min_number_of_days"]],
    [config["exclude_column"]],
    [config["save_preprocessed_file"]],
    [config["split_rate"]],
    [config["exclude_file"]],
  ):
    argument_dict = {
      "building_file": combo[0],
      "y_column": combo[1],
      "imputation_method": combo[2],
      "header": combo[3],
      "data_path": combo[4],
      "tmp_path": combo[5],
      "min_number_of_days": combo[6],
      "exclude_column": combo[7],
      "save_preprocessed_file": combo[8],
      "split_rate": combo[9],
      "exclude_file": combo[10],
    }
    preprocessing_arguments.append(argument_dict)

  return arguments, preprocessing_arguments, config