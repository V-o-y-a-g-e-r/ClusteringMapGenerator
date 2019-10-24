from time import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from python_research.experiments.unsupervised_segmentation.base_module import BaseModule
from python_research.experiments.unsupervised_segmentation.utils import MetricsEnum, get_metrics, save_generated_map, \
    save_gt_based_map


class RecurrentAutoencoder(BaseModule):

    def __init__(self, n_clusters: int, spatial_size: int, seq_len: int,
                 img_height: int, img_width: int, dest_path: str, gt: np.ndarray = None):
        super(RecurrentAutoencoder, self).__init__()
        if gt is not None:
            assert isinstance(gt, np.ndarray)
        self.img_height = img_height
        self.img_width = img_width
        self.gt = gt
        self.dest_path = dest_path
        self.gm = GaussianMixture(n_components=n_clusters)
        self.encoder = nn.GRU(input_size=spatial_size ** 2,
                              hidden_size=24,
                              num_layers=4,
                              batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=24, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=spatial_size ** 2 * seq_len)
        )
        self.cost_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.parameters())

    def forward(self, batch):
        isinstance(batch, torch.Tensor)
        batch = self._encode(batch)
        batch = self._decode(batch)
        return batch

    def eval_epoch(self, dataloader: DataLoader, metrics: dict, epoch: int):
        assert isinstance(dataloader, DataLoader)
        eval_time = time()
        self.eval()
        outputs, indexes = [], []
        with torch.no_grad():
            for batch, idx in tqdm(dataloader, total=len(dataloader)):
                if torch.cuda.is_available():
                    batch = batch.type(torch.cuda.FloatTensor)
                encoder_output = self._encode(batch)
                outputs.append(encoder_output.detach().cpu().numpy().squeeze())
                indexes.append(idx)
        outputs, indexes = np.vstack(outputs), np.vstack(indexes)
        print('Clustering...')
        self.gm.fit(outputs)
        clusters = self.gm.predict(outputs)
        gen_map = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        for y_pred, idx in zip(clusters, indexes):
            gen_map[idx[0], idx[1]] = y_pred
        metrics[MetricsEnum.EVAL_TIME].append(time() - eval_time)
        save_generated_map(gen_map, self.dest_path, epoch)
        if self.gt is not None:
            save_gt_based_map(gen_map, self.gt.copy(), self.dest_path, epoch)
            get_metrics(gt_hat=gen_map, gt=self.gt.copy(), metrics=metrics)

    def train_epoch(self, dataloader: DataLoader, metrics: dict):
        assert isinstance(dataloader, DataLoader)
        train_time = time()
        self.train()
        epoch_loss = []
        for batch, _ in tqdm(dataloader, total=len(dataloader)):
            if torch.cuda.is_available():
                batch = batch.type(torch.cuda.FloatTensor)
            self.zero_grad()
            out = self.forward(batch)
            loss = self.cost_function(out, batch.view(batch.shape[0], -1))
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.clone().detach().cpu().numpy())
        epoch_loss = np.mean(epoch_loss)
        print("Training MSE -> {}".format(epoch_loss))
        metrics[MetricsEnum.TRAIN_TIME].append(time() - train_time)
        metrics[MetricsEnum.MSE_LOSS].append(epoch_loss)

    def predict_epoch(self, *args, **kwargs):
        pass

    def _encode(self, batch: torch.Tensor):
        batch = self.encoder(batch)[0][:, -1, :]
        return batch

    def _decode(self, batch: torch.Tensor):
        batch = self.decoder(batch)
        return batch
