import yaml
import torch


def get_config(args):
    with open(args.config, 'r') as file:
        return yaml.safe_load(file)


def move_to_device(a_batch, device_is_cpu: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = a_batch
    if device_is_cpu:
        return x.to('cpu'), y.to('cpu')
    else:
        return x.to('cuda:0'), y.to('cuda:0')


def save_model(model, criteria, scheduler, loss, epoch):
    pass


def load_model(checkpoint_path):
    pass
