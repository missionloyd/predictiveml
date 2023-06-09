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
                **arg  # Merge all other key-value pairs from arg dictionary
            }
            updated_arguments.append(updated_arg)

    return updated_arguments
