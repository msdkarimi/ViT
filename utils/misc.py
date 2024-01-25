import yaml
import torch


def get_config(args):
    with open(args.config, 'r') as file:
        return yaml.safe_load(file)


def move_to_device(a_batch, device_is_cpu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = a_batch
    device = 'cpu' if device_is_cpu else 'cuda:0'
    return x.to(device), y.to(device)

