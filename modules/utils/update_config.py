
def update_config(config, cli_args):
    for key, value in cli_args.items():
        if value is not None and isinstance(value, bool):
            config[key] = value
        elif value is not None and not isinstance(value, bool):
            config[key] = [value]

    return config