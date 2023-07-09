import csv
from itertools import product  
from .save_results import save_winners_in, save_winners_out 

def save_args(results_file_path, in_file_path, out_file_path, config):
    args = []

    save_winners_in(results_file_path, in_file_path, config)
    
    with open(f'{in_file_path}/_winners.in', mode='r') as results_file:
        csv_reader = csv.DictReader(results_file)
        
        for row in csv_reader:
            argument = generate_arg(row, config)
            args.append(argument)

    save_winners_out(out_file_path, args, config)
            
    return

# Generate a list of arguments for model training
def generate_arg(row, config):
    argument = {
        "bldgname": str(row["bldgname"]),
        "model_data_path": str(row["model_data_path"]),
        "building_file": str(row["building_file"]),
        "y_column": str(row["y_column"]),
        "imputation_method": str(row["imputation_method"]),
        "model_type": str(row["model_type"]),
        "feature_method": str(row["feature_method"]),
        "n_feature": int(row["n_feature"]),
        "updated_n_feature": int(row["updated_n_feature"]),
        "time_step": int(row["time_step"]),
        "datelevel": str(row["datelevel"]),
        "header": list(config["header"]),
        "data_path": str(config["data_path"]),
        "add_feature": list(config["add_feature"]),
        "exclude_column": list(config["exclude_column"]),
        "n_fold": int(config["n_fold"]),
        "train_test_split": float(config["train_test_split"]),
        "minutes_per_model": int(config["minutes_per_model"]),
        "memory_limit": int(config["memory_limit"]),
        "save_model_file": True,  # Update with the desired value
        "save_model_plot": bool(config["save_model_plot"]),
        "path": str(config["path"]),
        "save_preprocessed_file": bool(config["save_preprocessed_file"]),
    }

    return argument
