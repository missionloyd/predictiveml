import pandas as pd
import numpy as np

def adaptive_sampling(arguments, path, temperature, n_args):
    # Load the results.csv file into a DataFrame
    df_results = pd.read_csv(path)

    # Sort the results based on the mape score in ascending order (lower is better)
    df_results_sorted = df_results.sort_values(by='mape', ascending=True)

    if len(df_results_sorted) > 0:
      # Calculate the number of results to consider for exploration and exploitation
      n_explore = int(n_args * (temperature / 2))
      n_exploit = n_explore
      # num_exploit = num_results - num_explore

      # Extract the top-performing combinations for exploitation
      exploitation_data = df_results_sorted.head(n_exploit)

      # Randomly select results for exploration that are not in the head
      exploration_data = df_results_sorted[~df_results_sorted.index.isin(exploitation_data.index)]
      exploration_data = exploration_data.sample(n=n_explore)

      # Combine the exploitation and exploration data
      arguments = pd.concat([exploitation_data, exploration_data])

    return arguments
