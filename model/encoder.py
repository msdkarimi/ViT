import torch.nn as nn
from model import register


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        pass

    def forward(self, ):
        pass


@register
def get_encoder(config):
    return Encoder(config)
