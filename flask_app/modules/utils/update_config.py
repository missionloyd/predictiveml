from modules.utils.create_results_file_path import create_results_file_path

def update_config(config, cli_args):
    for key, value in cli_args.items():
        if value is not None and isinstance(config[key], list) and not isinstance(value, list):
            config[key] = [value]
        elif value is not None and not isinstance(config[key], list) and not isinstance(value, list):
            config[key] = value

            if key == 'results_file':
                config['results_file_path'] = create_results_file_path(config['path'], config['results_file'])

    return config