from config.args import args
import yaml


def get_config():
    with open(args.config, 'r') as file:
        return yaml.safe_load(file)
