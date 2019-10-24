import os
from enum import Enum

import torch
from torch.utils.data.dataloader import DataLoader

from  python_research.experiments.unsupervised_segmentation.base_module import BaseModule


class MetricsEnum(Enum):
    MSE_LOSS = "mse_loss"
    NMI_SCORE = "nmi_score"
    ARS_SCORE = "ars_score"
    TRAIN_TIME = "train_time"
    EVAL_TIME = "eval_time"


class Pipeline(object):

    def __init__(self, model: BaseModule, dataloader: DataLoader, epochs: int, patience: int):
        assert isinstance(model, BaseModule)
        assert isinstance(dataloader, DataLoader)
        assert hasattr(model, 'dest_path')
        assert hasattr(model, 'optimizer')
        self.metrics = {
            MetricsEnum.MSE_LOSS: [],
            MetricsEnum.NMI_SCORE: [],
            MetricsEnum.ARS_SCORE: [],
            MetricsEnum.TRAIN_TIME: [],
            MetricsEnum.EVAL_TIME: [],
        }
        self.model = model
        self.dataloader = dataloader
        self.epochs = epochs
        self.patience = patience

    def stopping_condition(self) -> bool:
        """
        Return True if no improvement and should stop learning
        :return: Boolean
        """
        return min(self.metrics[MetricsEnum.MSE_LOSS][:-self.patience]) <= \
               min(self.metrics[MetricsEnum.MSE_LOSS][-self.patience:])

    def save_model(self, epoch) -> bool:
        saved_model_flag = False
        if len(self.metrics[MetricsEnum.MSE_LOSS]) < 2 or \
                self.metrics[MetricsEnum.MSE_LOSS][-1] < min(self.metrics[MetricsEnum.MSE_LOSS][:-1]):
            print("Saving improvement...")
            torch.save({
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
                MetricsEnum.MSE_LOSS: self.metrics[MetricsEnum.MSE_LOSS][-1],
            }, os.path.join(self.model.dest_path, "epoch-{}-checkpoint".format(epoch)))
            saved_model_flag = True
        return saved_model_flag

    def run_pipeline(self):
        for epoch in range(self.epochs):
            print("Epoch -> {}".format(epoch))
            self.model.train_epoch(self.dataloader, self.metrics)
            if epoch > self.patience and self.stopping_condition():
                break
            if self.save_model(epoch):
                self.model.eval_epoch(self.dataloader, self.metrics, epoch)
            print(self.metrics)
