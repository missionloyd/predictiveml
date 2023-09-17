
def preprocess(config):
  from modules.utils.logging_config import logging_config
  from modules.utils.create_args import create_args
  from modules.utils.adaptive_sampling import adaptive_sampling
  from modules.utils.process_batch_args import process_batch_args
  from modules.utils.match_args import match_args
  from modules.utils.save_results import save_preprocessed_args_results
  from modules.preprocessing_methods.main import preprocessing

  batch_size = config['batch_size']
  n_jobs = config['n_jobs']
  args_file_path = config['args_file_path']

  logging_config()
  args, preprocess_args = create_args(config)
  args = adaptive_sampling(args, config)

  # Process the preprocessing arguments
  processed_args = process_batch_args('Preprocessing', preprocess_args, preprocessing, batch_size, n_jobs, config)
  
  # Match processed columns with original argument combinations
  updated_args = match_args(args, processed_args)

  # Save the args to the CSV file
  save_preprocessed_args_results(args_file_path, updated_args)

  return