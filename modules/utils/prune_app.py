import shutil, os, csv

def prune():

    directories_to_prune = [
      'logs/api_log',
      'logs/debug_log',
      'logs/error_log',
      'logs/flask_log',
      'logs/info_log',
      'models/tmp',
      'models/ensembles',
      'models/solos',
      'models/xgboost',
    ]

    for i in range(0, len(directories_to_prune)):
        print(directories_to_prune[i])
        if os.path.exists(directories_to_prune[i]):
            shutil.rmtree(directories_to_prune[i])
            os.makedirs(directories_to_prune[i])
                    
        else:
            os.makedirs(directories_to_prune[i])
    return