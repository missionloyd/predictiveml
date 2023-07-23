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
    [config["save_model_file"]],
    [config["updated_n_feature"]],
    [config["selected_features_delimited"]],
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
      "save_model_file": combo[8],
      "updated_n_feature": combo[9],
      "selected_features_delimited": combo[10],
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
    [config["exclude_column"]],
    [config["save_preprocessed_file"]],
    [config["train_test_split"]],
    [config["train_ratio_threshold"]],
    [config["test_ratio_threshold"]],
    [config["exclude_file"]],
  ):
    argument_dict = {
      "building_file": combo[0],
      "y_column": combo[1],
      "imputation_method": combo[2],
      "header": combo[3],
      "data_path": combo[4],
      "tmp_path": combo[5],
      "exclude_column": combo[6],
      "save_preprocessed_file": combo[7],
      "train_test_split": combo[8],
      "train_ratio_threshold": combo[9],
      "test_ratio_threshold": combo[10],
      "exclude_file": combo[11],
    }
    preprocessing_arguments.append(argument_dict)

  return arguments, preprocessing_arguments, config