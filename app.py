import subprocess
import threading
import os, time
from flask import Flask, url_for, jsonify, request, render_template
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app, origins='http://localhost:3000')
socketio = SocketIO(app)

n_jobs_counter = 0
lock = threading.Lock()

previous_results = {}  # Dictionary to store previous prediction results

def run_main(flags):
    global n_jobs_counter
    with lock:
        n_jobs_counter += 1
        job_id = n_jobs_counter
        command_main = ['python3', '-u', 'main.py'] + flags + ['--job_id', str(job_id)]

    def run_job():
        print(f'Starting job execution for job ID: {job_id}')
        debug_log_file = f'logs/debug_log/{job_id}.log'
        error_log_file = f'logs/error_log/{job_id}.log'

        with open(debug_log_file, 'w') as debug_log, open(error_log_file, 'w') as error_log:
            debug_log.write("")
            error_log.write("")

        def emit_info_log_lines():
            info_log_file = f'logs/info_log/{job_id}.log'

            # Wait for the file to become available
            while not os.path.isfile(info_log_file):
                time.sleep(0.1)  # Sleep for 1 second

            process = subprocess.Popen(
                ['tail', '-f', info_log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line-buffered output
                universal_newlines=True  # Decode output as text
            )

            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                socketio.emit('new_info_log_line', {'job_id': job_id, 'line': line})

            process.wait()  # Wait for the process to complete

        # Start the info log emission in a separate thread
        info_job_thread = threading.Thread(target=emit_info_log_lines)
        info_job_thread.start()

        with open(debug_log_file, 'w') as debug_log, open(error_log_file, 'w') as error_log:
            process = subprocess.Popen(
                command_main,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line-buffered output
                universal_newlines=True  # Decode output as text
            )

            for line in process.stdout:  # Read stdout line by line in real-time
                socketio.emit('new_debug_log_line', {'job_id': job_id, 'line': line.strip()})
                debug_log.write(line)

            for line in process.stderr:  # Read stderr line by line in real-time
                socketio.emit('new_error_log_line', {'job_id': job_id, 'line': line.strip()})
                error_log.write(line)

        process.wait()  # Wait for the process to complete

    # Start the job in a separate thread
    job_thread = threading.Thread(target=run_job)
    job_thread.start()

    return job_id

def run_api_log(api_file):
    with open(api_file, 'r') as file:
        data = eval(file.read())
    return data

def run_info_log(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]  # Remove leading/trailing spaces
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

    while not os.path.isfile(f'logs/info_log/{job_id}.log'):
        time.sleep(0.1)  # Sleep for 1 second

    links = '<br>'.join([
        f'',
        f'<a href="{url_for("display_info_log", job_id=job_id)}">Info Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_debug_log", job_id=job_id)}">Debug Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_error_log", job_id=job_id)}">Error Job Log: #{job_id}</a>',
        f'',
    ])
    return links

@app.route('/')
def display_app_log():
    job_id = 0
    log_directory = "logs/"

    # Create the log directories if they don't exist
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(log_directory + "debug_log", exist_ok=True)
    os.makedirs(log_directory + "error_log", exist_ok=True)
    os.makedirs(log_directory + "flask_log", exist_ok=True)
    os.makedirs(log_directory + "info_log", exist_ok=True)
    os.makedirs(log_directory + "api_log", exist_ok=True)

    # Create the log files if they don't exist
    log_files = {
        "debug_log": f'{log_directory}debug_log/0.log',
        "error_log": f'{log_directory}error_log/0.log',
        "flask_log": f'{log_directory}flask_log/0.log',
        "info_log": f'{log_directory}info_log/0.log'
    }

    for log_type, log_file in log_files.items():
        open(log_file, 'w').close()

    log_content = run_info_log(log_files["flask_log"])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id=job_id)

    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/demo/predict/<string:building_file>/<string:y_column>')
def run_predict_demo(building_file, y_column):
    job_id = run_main(['--predict', '--building_file', building_file, '--y_column', y_column, '--time_step', '1', '--datelevel', 'hour'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    log_content = f'Job submitted (--predict --building_file {building_file} --y_column {y_column} --time_step 1 --datelevel hour --job_id {job_id})'
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)


# @app.route('/predict/<string:building_file>/<string:y_column>')
# def run_predict(building_file, y_column):
#     # Create a key tuple from the input arguments
#     key = (building_file, y_column)

#     # Check if the key exists in the previous results
#     if key in previous_results:
#         api_file = previous_results[key]
#         data = run_api_log(api_file)
#         return jsonify({'job_id': 0, 'data': data, 'status': 'ok'})

#     # No previous result found, proceed with running the prediction
#     job_id = run_main(['--predict', '--building_file', building_file, '--y_column', y_column, '--time_step', '1', '--datelevel', 'hour'])
#     api_file = f'logs/api_log/{job_id}.log'

#     # Wait for the API file to become available
#     while not os.path.isfile(api_file):
#         time.sleep(0.1)  # Sleep for 1 second

#     data = run_api_log(api_file)

#     # Save the result for future use
#     previous_results[key] = api_file

#     return jsonify({'job_id': job_id, 'data': data, 'status': 'ok'})

@app.route('/predict', methods=['POST'])
def run_forecast():
    # Extract parameters from the request body
    y_column = request.json.get('y_column')
    building_file = request.json.get('building_file')
    startDate = request.json.get('startDate') 
    endDate = request.json.get('endDate') 
    time_step = request.json.get('time_step')
    datelevel = request.json.get('datelevel') 
    table = request.json.get('table')

    # Create a key tuple from the input arguments
    key = (y_column, building_file, startDate, endDate, time_step, datelevel, table)

    # Check if the key exists in the previous results
    if key in previous_results:
        api_file = previous_results[key]
        data = run_api_log(api_file)
        return jsonify({'job_id': 0, 'data': data, 'status': 'ok'})

    # No previous result found, proceed with running the prediction
    job_id = run_main(['--predict', '--building_file', building_file, '--y_column', y_column, '--startDate', startDate, '--endDate', endDate, '--time_step', time_step, '--datelevel', datelevel, 'table', table])
    api_file = f'logs/api_log/{job_id}.log'

    # Wait for the API file to become available
    while not os.path.isfile(api_file):
        time.sleep(0.1)  # Sleep for 0.1 second

    data = run_api_log(api_file)

    # Save the result for future use
    previous_results[key] = api_file

    return jsonify({'job_id': job_id, 'data': data, 'status': 'ok'})


@app.route('/info_log/<int:job_id>')
def display_info_log(job_id):
    log_file = f'logs/info_log/{job_id}.log'
    log_content = run_info_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/debug_log/<int:job_id>')
def display_debug_log(job_id):
    log_file = f'logs/debug_log/{job_id}.log'
    log_content = run_info_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/error_log/<int:job_id>')
def display_error_log(job_id):
    log_file = f'logs/error_log/{job_id}.log'
    log_content = run_info_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/preprocess')
def run_preprocess():
    job_id = run_main(['--preprocess'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    log_content = f'Job submitted (--preprocess --job_id {job_id})'
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/preprocess_train')
def run_preprocess_train():
    job_id = run_main(['--preprocess', '--train', '--save', '--time_step', '1', '--datelevel', 'hour'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    log_content = f'Job submitted (--preprocess --train --save --time_step 1 --datelevel hour --job_id {job_id})'
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/train')
def run_train():
    job_id = run_main(['--train', '--save', '--time_step', '1', '--datelevel', 'hour'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    log_content = f'Job submitted (--train --save --time_step 1 --datelevel hour --job_id {job_id})'
    return render_template('index.html', command_links=command_links, log_links=log_links, log_content=log_content, job_id=job_id)

@app.route('/<string:log_type>/<path:subpath>/log_content')
def get_log_content(log_type, subpath):
    log_file = f'logs/{log_type}/{subpath}.log'
    log_content = run_info_log(log_file)    
    return log_content


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, threaded=True)