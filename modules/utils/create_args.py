import json
import pandas as pd
from itertools import product  
from modules.utils.adaptive_sampling import adaptive_sampling

# Generate a list of arguments for model training
def create_args(config):
    arguments = []

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
    ):
        argument_dict = {
            "building_file": combo[0],
            "y_column": combo[1],
            "imputation_method": combo[2],
        }
        preprocessing_arguments.append(argument_dict)


    return arguments, preprocessing_arguments
