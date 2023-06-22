import os, sys, json
import numpy as np

job_id = None
user_commands = None

def logger(message):
    global job_id  # Declare job_id as global

    print(message)

    log_directory = "logs/info_log/"
    log_file = f"{job_id}.log"
    log_path = os.path.join(log_directory, log_file)

    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Open the log file in append mode
    with open(log_path, "a") as file:
        file.write(f"{message}\n")

def api_logger(message):
    global job_id  # Declare job_id as global

    log_directory = "logs/api_log/"
    log_file = f"{job_id}.log"
    log_path = os.path.join(log_directory, log_file)

    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Convert ndarray to list
    if isinstance(message, np.ndarray):
        message = message.tolist()

    # Open the log file in append mode
    with open(log_path, "w") as file:
        json.dump(message, file)

def setup_logger(new_job_id):
    global user_commands
    global job_id  # Declare job_id as global
    job_id = new_job_id

    log_directory = "logs/info_log/"
    log_file = f"{job_id}.log"
    log_path = os.path.join(log_directory, log_file)

    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Open the log file in append mode
    with open(log_path, "w") as file:
        file.write(f'Executing Job ({user_commands}):\n')


def extract_args():
    global user_commands
    # Extract the arguments and their values into a string
    user_commands = ' '.join(sys.argv[1:])  # Exclude the script name (sys.argv[0])