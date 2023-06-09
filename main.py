#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from config import load_config
from modules.utils.create_args import create_args
from modules.utils.process_batch_args import process_batch_args
from modules.preprocessing_methods.main import preprocessing
from modules.utils.match_args import match_args
from modules.training_methods.main import train_model
from modules.utils.save_results import save_results
from modules.utils.calculate_duration import calculate_duration


# In[ ]:


if __name__ == '__main__':
  start_time = time.time()

  config = load_config()

  path = config['path']
  results_header = config['results_header']
  batch_size = config['batch_size']

  # Generate a list of arguments for model training
  args, preprocess_args = create_args(config)

  # fill in missing gaps in each y_column, add each updated column as an argument 
  processed_args = process_batch_args('Preprocessing', preprocess_args, batch_size, preprocessing)

  # match processed columns with original argument combinations
  updated_args = match_args(args, processed_args)
  
  # Execute the training function in parallel for each batch of arguments
  results = process_batch_args('Training', updated_args, batch_size, train_model)

  # Convert the results to a set to remove any duplicates
  unique_results = set(results)

  # Save the results to the CSV file
  save_results(path, results_header, unique_results)

  # calculate duration of script
  duration = calculate_duration(start_time)

  print(f"Success... Time elapsed: {duration} hours.")

