import sys
from config.args import args
from utils.misc import get_config
from experiments import Experiment


def main(config: dict):
    experiment = Experiment(config, args)
    print(experiment.model)


if __name__ == '__main__':
    config = get_config(args)
    main(config)
    sys.exit(0)
