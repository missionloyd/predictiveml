def match_args(arguments, preprocessed_columns):
    col_dict = {(col[0], col[1], col[2]): col for col_list in preprocessed_columns if col_list for col in col_list}

    updated_arguments = [(col[3], col[4], *arg) for arg in arguments if (col := col_dict.get((arg[0], arg[1], arg[2])))]
    return updated_arguments


# def match_args(arguments, preprocessed_columns):
#     col_dict = {}

#     for col_list in preprocessed_columns:
#         if col_list is not None:
#             col = col_list[0]
#             col_dict[(col[0], col[1], col[2])] = col

#     updated_arguments = []
#     for arg in arguments:
#         col = col_dict.get((arg[0], arg[1], arg[2]))
#         if col is not None:
#             model_data_path = col[3]
#             bldgname = col[4]
#             updated_arg = (model_data_path, bldgname, *arg)
#             updated_arguments.append(updated_arg)

#     return updated_arguments
