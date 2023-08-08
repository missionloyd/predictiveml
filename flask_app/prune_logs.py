import shutil, os, argparse
from modules.logging_methods.main import extract_args

def prune_logs(job_id):    
    directories_to_prune = [
      'logs/api_log',
      'logs/debug_log',
      'logs/error_log',
      'logs/flask_log',
      'logs/info_log',
    #   'models/tmp',
    #   'models/ensembles',
    #   'models/solos',
    #   'models/xgboost',
    ]

    for i in range(0, len(directories_to_prune)):
        if os.path.exists(directories_to_prune[i]):
            shutil.rmtree(directories_to_prune[i])
            os.makedirs(directories_to_prune[i])
                    
        else:
            os.makedirs(directories_to_prune[i])

        with open(f'{directories_to_prune[i]}/{job_id}.log', 'w') as log_file:
            log_file.write("")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for preprocessing, training, or predicting.')
    parser.add_argument('--job_id', type=int, help='Job ID for logging.')
    args = parser.parse_args()
    extract_args()
    job_id = args.job_id
    prune_logs(job_id)