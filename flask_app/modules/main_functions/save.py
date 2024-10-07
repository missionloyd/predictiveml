
def save(config):
  from modules.utils.process_batch_args import process_batch_args
  from modules.utils.load_args import load_args
  from modules.utils.save_args import save_args
  from modules.prediction_methods.on_demand_prediction import on_demand_prediction
  from modules.training_methods.main import train_model

  batch_size = config['batch_size']
  n_jobs = config['n_jobs']
  results_file_path = config['results_file_path']
  winners_in_file_path = config['winners_in_file_path']
  winners_out_file_path = config['winners_out_file_path']
  winners_out_file = config['winners_out_file']

  save_args(results_file_path, winners_in_file_path, winners_out_file_path, config)
  updated_args = load_args(winners_out_file)

  # Re-train winners and save the model file
  results = process_batch_args('Saving', updated_args, train_model, batch_size, n_jobs, config)
                
  return