from model import ViT
from model_builder import Registry
from utils import get_config
import torch


def main(cfg: dict):
    model = Registry.from_conf_create_model(cfg['MODEL']['NAME'], config=cfg['MODEL'])
    print(model)


if __name__ == '__main__':
    cfg = get_config()
    main(cfg)
