import subprocess
import threading
from flask import Flask, url_for, jsonify, request

app = Flask(__name__)

n_jobs_counter = 0
lock = threading.Lock()

def run_main(flags):
    global n_jobs_counter
    with lock:
        n_jobs_counter += 1
        job_id = n_jobs_counter

    def run_job():
        main_log_file = f'logs/main_log/{job_id}.log'
        error_log_file = f'logs/error_log/{job_id}.log'

        with open(main_log_file, 'w') as main_log, open(error_log_file, 'w') as error_log:
            main_log.write("")
            error_log.write("")

        command_main = ['python3', '-u', 'main.py'] + flags  # Add '-u' flag for unbuffered output
        with open(main_log_file, 'w') as main_log, open(error_log_file, 'w') as error_log:
            process = subprocess.Popen(
                command_main,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line-buffered output
                universal_newlines=True  # Decode output as text
            )

            for line in process.stdout:  # Read stdout line by line in real-time
                main_log.write(line)

            for line in process.stderr:  # Read stderr line by line in real-time
                error_log.write(line)

        process.wait()  # Wait for the process to complete

    # Start the job in a separate thread
    job_thread = threading.Thread(target=run_job)
    job_thread.start()

    return job_id

def run_main_log(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    return '<br>'.join(lines)

def generate_command_links():
    links = '<br>'.join([
        f'<a href="/">Home</a>',
        f'',
        f'<a href="{url_for("run_preprocess")}">Run New Preprocessing Job</a>',
        f'<a href="{url_for("run_train")}">Run New Training Job</a>',
        f'<a href="{url_for("run_preprocess_train")}">Run New Preprocessing + Training Job</a>',
        f'<a href="{url_for("run_predict_demo", building_file="Stadium_Data_Extended.csv", y_column="present_elec_kwh")}">Run New Demo Prediction Job</a>',
    ])
    return links

def generate_log_links(job_id):
    links = '<br>'.join([
        f'',
        f'<a href="{url_for("display_main_log", job_id=job_id)}">View Main Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_error_log", job_id=job_id)}">View Error Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_flask_log", job_id=job_id)}">View App Log</a>'
        f'',
        f'',
    ])
    return links

@app.route('/')
def display_app_log():
    log_file = f'logs/flask_log/0.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id=0)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/predict/<string:building_file>/<string:y_column>')
def run_predict_demo(building_file, y_column):
    # Perform any necessary operations using the received arguments
    job_id = run_main(['--predict', '--building_file', building_file, '--y_column', y_column])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted (--predict --building_file Stadium_Data_Extended.csv --y_column present_elec_kwh): #{job_id}"

@app.route('/main_log/<int:job_id>')
def display_main_log(job_id):
    log_file = f'logs/main_log/{job_id}.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/error_log/<int:job_id>')
def display_error_log(job_id):
    log_file = f'logs/error_log/{job_id}.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/flask_log/<int:job_id>')
def display_flask_log(job_id):
    log_file = f'logs/flask_log/0.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/preprocess')
def run_preprocess():
    job_id = run_main(['--preprocess'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted (--preprocess): #{job_id}"

@app.route('/preprocess_train')
def run_preprocess_train():
    job_id = run_main(['--preprocess', '--train', '--save'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted (--preprocess --train --save): #{job_id}"

@app.route('/train')
def run_train():
    job_id = run_main(['--train', '--save'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted (--train --save): #{job_id}"

# Unused at the moment
# @app.route('/predict')
# def run_predict():
#     # Get the JSON data from the request body
#     data = request.get_json()
    
#     # Extract the required arguments from the data
#     building_file = data.get('building_file')
#     y_column = data.get('y_column')
#     # bldgname = data.get('bldgname')
#     # startDate = data.get('startDate')
#     # endDate = data.get('endDate')
#     # datelevel = data.get('datelevel')
#     # table = data.get('table')

#     # Perform any necessary operations using the received arguments
#     y_pred = run_main(['--predict', '--building_file', building_file, '--y_column', y_column])

#     # Create a response dictionary
#     response = {
#         data: y_pred,
#         'status': 'ok',
#     }

#     # Return the JSON response
#     return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
