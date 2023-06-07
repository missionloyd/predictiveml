def match_args(arguments, preprocessed_columns):
    # Create a dictionary of (bldgname, col1, col2) tuples as keys and corresponding col data as values
    col_dict = {(col["building_file"], col["y_column"], col["imputation_method"]): col for col_list in preprocessed_columns if col_list for col in col_list}

    updated_arguments = []
    for arg in arguments:
        # Retrieve the corresponding col data from col_dict using (bldgname, col1, col2) as the key
        col = col_dict.get((arg["building_file"], arg["y_column"], arg["imputation_method"]))
        if col:

            # Create the updated argument dictionary with the required values
            updated_arg = {
                "model_data_path": col["model_data_path"],
                "bldgname": col["bldgname"],
                "building_file": arg["building_file"],
                "y_column": arg["y_column"],
                "imputation_method": arg["imputation_method"],
                "model_type": arg["model_type"],
                "feature_method": arg["feature_method"],
                "n_feature": arg["n_feature"],
                "time_step": arg["time_step"],
                "header": arg["header"],
                "data_path": arg["data_path"],
                "add_feature": arg["add_feature"],
                "min_number_of_days": arg["min_number_of_days"],
                "exclude_column": arg["exclude_column"],
                "n_fold": arg["n_fold"],
                "split_rate": arg["split_rate"],
                "minutes_per_model": arg["minutes_per_model"],
                "memory_limit": arg["memory_limit"],
                "save_model_file": arg["save_model_file"],
                "save_model_plot": arg["save_model_plot"],
                "path": arg["path"]
            }
            updated_arguments.append(updated_arg)

    return updated_arguments


# def match_args(arguments, preprocessed_columns):
#     col_dict = {(col[0], col[1], col[2]): col for col_list in preprocessed_columns if col_list for col in col_list}

#     updated_arguments = [(col[3], col[4], *arg) for arg in arguments if (col := col_dict.get((arg[0], arg[1], arg[2])))]
#     return updated_arguments
