import sys
from model import build_model
from utils import get_config


def main(config: dict):
    model = build_model(config, config['MODEL']['NAME'])
    print(model)


if __name__ == '__main__':
    config = get_config()
    main(config)
    sys.exit(0)
