import json
import pandas as pd
from itertools import product  
from modules.preprocessing_methods.adaptive_sampling import adaptive_sampling

# Generate a list of arguments for model training
def create_args(config, cli_args):
    arguments = []
    
    results_file_path = config['results_file_path']

    if cli_args['building_file']:
        config['building_file'] = [cli_args['building_file']]

    if cli_args['time_step'] and cli_args['datelevel']:
        config['time_step'] = [cli_args['time_step']]
        config['datelevel'] = [cli_args['datelevel']]

    if cli_args['temperature']:
        config['temperature'] = cli_args['temperature']


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

    # Assuming arguments is a DataFrame or can be converted to a DataFrame
    # (It is better to ensure that arguments is a DataFrame before calling this function)
    temperature = config['temperature']

    if temperature > 0 and temperature < 1:
        arguments = pd.DataFrame(data=arguments)

        n_args = len(arguments)

        filtered_data = adaptive_sampling(arguments, results_file_path, temperature, n_args)

        if not filtered_data.empty:
            # Add the additional columns to the DataFrame directly
            filtered_data["save_model_file"] = config["save_model_file"]
            filtered_data["updated_n_feature"] = config["updated_n_feature"]
            filtered_data["selected_features_delimited"] = config["selected_features_delimited"]

            # Extract the relevant arguments from filtered_data as a list of dictionaries
            arguments = [dict(row) for _, row in filtered_data.iterrows()]

    else:
        # Convert the data to a DataFrame (if it's not already)
        arguments_df = pd.DataFrame(data=arguments)

        if not arguments_df.empty:
            # Extract the relevant arguments from arguments_df as a list of dictionaries
            arguments = [dict(row) for _, row in arguments_df.iterrows()]


    return arguments, preprocessing_arguments, config
