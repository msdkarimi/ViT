import torch.nn as nn
from copy import deepcopy
from abc import ABCMeta, abstractmethod


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('somthing important called forward pass has been excluded!')
