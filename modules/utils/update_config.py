
def update_config(config, cli_args):
    for key, value in cli_args.items():
        if value is not None:
            config[key] = [value]

    return config