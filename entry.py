import sys
from model import get_vit
from utils import get_config


def main(config: dict):
    model = get_vit(config)
    print(model)


if __name__ == '__main__':
    config = get_config()
    main(config)
    sys.exit(0)
