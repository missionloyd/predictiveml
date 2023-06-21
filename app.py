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
        main_log_file = f'/app/main_log/{job_id}.log'
        error_log_file = f'/app/error_log/{job_id}.log'

        command_main = ['python3', 'main.py'] + flags
        with open(main_log_file, 'w') as main_log, open(error_log_file, 'w') as error_log:
            process = subprocess.Popen(command_main, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            main_log.write(stdout.decode('utf-8'))
            error_log.write(stderr.decode('utf-8'))

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
        # f'<a href="{url_for("run_predict")}">Run Prediction Job</a><br>',
        f'<a href="{url_for("run_preprocess")}">Run Preprocessing Job</a>',
        f'<a href="{url_for("run_preprocess_train")}">Run Preprocessing + Training Job</a>',
        f'<a href="{url_for("run_train")}">Run Training Job</a>'
    ])
    return links

def generate_log_links(job_id):
    links = '<br>'.join([
        f'',
        f'<a href="{url_for("display_main_log", job_id=job_id)}">View Main Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_error_log", job_id=job_id)}">View Error Job Log: #{job_id}</a>',
        f'<a href="{url_for("display_flask_log", job_id=job_id)}">View App Log</a>'
    ])
    return links

@app.route('/')
def display_app_log():
    log_file = f'/app/flask_log/0.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id=0)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/predict')
def run_predict():
    # Get the JSON data from the request body
    data = request.get_json()
    
    # Extract the required arguments from the data
    bldgname = data.get('bldgname')
    # startDate = data.get('startDate')
    # endDate = data.get('endDate')
    # datelevel = data.get('datelevel')
    # table = data.get('table')

    # Perform any necessary operations using the received arguments
    y_pred = run_main(['--predict', '--bldgname', bldgname])

    # Create a response dictionary
    response = {
        data: y_pred,
        'status': 'ok',
    }

    # Return the JSON response
    return jsonify(response)
@app.route('/main_log/<int:job_id>')
def display_main_log(job_id):
    log_file = f'/app/main_log/{job_id}.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/error_log/<int:job_id>')
def display_error_log(job_id):
    log_file = f'/app/error_log/{job_id}.log'
    log_content = run_main_log(log_file)
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br>{log_content}"

@app.route('/flask_log/<int:job_id>')
def display_flask_log(job_id):
    log_file = f'/app/flask_log/0.log'
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
    job_id = run_main(['--preprocess', '--train'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted ('--preprocess --train'): #{job_id}"

@app.route('/train')
def run_train():
    job_id = run_main(['--train'])
    command_links = generate_command_links()
    log_links = generate_log_links(job_id)
    return f"{command_links}<br>{log_links}<br><br>Job submitted ('--train'): #{job_id}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
