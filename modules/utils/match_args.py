def match_args(arguments, preprocessed_columns):
    # Create a dictionary of (bldgname, col1, col2) tuples as keys and corresponding col data as values
    col_dict = {(col[0], col[1], col[2]): col for col_list in preprocessed_columns if col_list for col in col_list}

    updated_arguments = []
    for arg in arguments:
        # Retrieve the corresponding col data from col_dict using (bldgname, col1, col2) as the key
        col = col_dict.get((arg[0], arg[1], arg[2]))
        if col:
            # Extract required values from the col data
            model_data_path = col[3]
            bldgname = col[4]
            n_features = arg[5]
            add_features = arg[9]

            # Count the number of selected features available in the col data
            num_selected_features = 0
            for feature in add_features:
                if (bldgname, arg[1], arg[2], feature) in col_dict:
                    num_selected_features += 1

            # Update n_features if it exceeds the number of selected features
            if n_features >= num_selected_features:
                n_features = num_selected_features

            # Create the updated argument tuple with the required values
            updated_arg = (model_data_path, bldgname, *arg[1:5], n_features, *arg[6:])
            updated_arguments.append(updated_arg)

    return updated_arguments


# def match_args(arguments, preprocessed_columns):
#     col_dict = {(col[0], col[1], col[2]): col for col_list in preprocessed_columns if col_list for col in col_list}

#     updated_arguments = [(col[3], col[4], *arg) for arg in arguments if (col := col_dict.get((arg[0], arg[1], arg[2])))]
#     return updated_arguments
