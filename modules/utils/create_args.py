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
    [config["exclude_column"]],
    [config["n_fold"]],
    [config["train_test_split"]],
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
      "exclude_column": combo[11],
      "n_fold": combo[12],
      "train_test_split": combo[13],
      "minutes_per_model": combo[14],
      "memory_limit": combo[15],
      "save_model_file": combo[16],
      "save_model_plot": combo[17],
      "path": combo[18],
      "save_preprocessed_file": combo[19],
      "updated_n_feature": combo[20],
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