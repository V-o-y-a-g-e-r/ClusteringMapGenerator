from abc import ABC, abstractmethod

from torch import nn


class BaseModule(ABC, nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_epoch(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval_epoch(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict_epoch(self, *args, **kwargs):
        pass
