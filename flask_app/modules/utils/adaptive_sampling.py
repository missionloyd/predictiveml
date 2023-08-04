import pandas as pd
import numpy as np

def adaptive_sampling(args, config):
    args = pd.DataFrame(data=args)
    results_file_path = config['results_file_path']
    target_error = config['target_error']
    temperature = config['temperature']

    if temperature > 0 and temperature < 1:
        df_results = pd.read_csv(results_file_path)
        df_results_sorted = df_results.sort_values(by=target_error, ascending=True)
        print(len(df_results_sorted))

        if len(df_results_sorted) > 0:
            key = ['y_column', 'building_file', 'datelevel', 'time_step']
            results_grouped_data = df_results_sorted.groupby(key)
            args_grouped_data = args.groupby(key)

            exploitation_data_list = []
            exploration_data_list = []

            for _, group_df in results_grouped_data:
                group_key = tuple(group_df[key].iloc[0])
                n_args_group = len(args_grouped_data.get_group(group_key))

                n_explore = max(int(len(group_df) * (temperature / 2)), int(n_args_group * (temperature / 2)), 1)
                n_exploit = n_explore

                exploit_group_data = group_df.head(n_exploit)
                exploitation_data_list.append(exploit_group_data)

                explore_group_data = group_df[~group_df.index.isin(exploit_group_data.index)]

                if len(explore_group_data) > 0:
                    n_explore = min(n_explore, len(explore_group_data))
                    explore_group_data = explore_group_data.sample(n=n_explore, replace=False)
                    exploration_data_list.append(explore_group_data)

            exploitation_data = pd.concat(exploitation_data_list)
            exploration_data = pd.concat(exploration_data_list)

            args = pd.concat([exploitation_data, exploration_data])

            if not args.empty:
                # Add the additional columns to the DataFrame directly
                args["save_model_file"] = config["save_model_file"]
                args["updated_n_feature"] = config["updated_n_feature"]
                args["selected_features_delimited"] = config["selected_features_delimited"]

    args = [dict(row) for _, row in args.iterrows()]

    return args
