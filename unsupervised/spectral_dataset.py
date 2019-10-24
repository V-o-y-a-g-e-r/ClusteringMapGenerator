import math

import numpy as np
import torch
from tifffile import imread
from torch.utils.data import dataset, dataloader


class SpectralDataset(dataset.Dataset):
    SPECTRAL_AXIS = -1
    ROW_AXIS = 0
    COLUMN_AXIS = 1
    SHUFFLE = True

    def __init__(self, spatial_size: int):
        super(SpectralDataset, self).__init__()
        self.spatial_size = spatial_size
        self.pad_size = int((self.spatial_size - 1) / 2)
        self.data_cube = None
        self.ground_truth = None
        self.spectral_size = None
        self.row_size = None
        self.column_size = None
        self.dataloader = None

    def __len__(self) -> int:
        return int(self.row_size * self.column_size)

    def __getitem__(self, index: int) -> list:
        row_index, col_index = int(index / self.column_size) + self.pad_size, \
                               int(index % self.column_size) + self.pad_size
        sample = None
        if self.spatial_size == 1:
            sample = torch.FloatTensor(self.data_cube[row_index, col_index]).unsqueeze(dim=-1)
        elif self.spatial_size > 1:
            sample = torch.FloatTensor(self.data_cube[row_index - self.pad_size:row_index + self.pad_size + 1,
                                       col_index - self.pad_size:col_index + self.pad_size + 1, :]).permute(2, 0, 1)
            sample = sample.view(sample.shape[0], -1)
            row_index, col_index = row_index - self.pad_size, col_index - self.pad_size
        return [sample, torch.LongTensor(np.array([row_index, col_index]))]

    def get_dataloader(self):
        return dataloader.DataLoader(dataset=self, batch_size=4489,
                                     shuffle=self.__class__.SHUFFLE, num_workers=1)

    def load_data(self, data_path: str, gt_path: str = None):
        try:
            if data_path.endswith('.npy'):
                self.data_cube = np.load(data_path).astype(float)
            elif data_path.endswith('.tif'):
                self.data_cube = imread(data_path).astype(float)
            else:
                raise ValueError("This type of file is not supported for loading.")
        except (IOError, ValueError) as error:
            print(error)
        if self.spatial_size > 1:
            self.data_cube = np.pad(array=self.data_cube,
                                    pad_width=((self.pad_size, self.pad_size),
                                               (self.pad_size, self.pad_size),
                                               (0, 0)),
                                    mode="constant")
        if gt_path is not None:
            if gt_path.endswith('.npy'):
                self.ground_truth = np.load(gt_path).astype(np.uint8)
        self.spectral_size = self.data_cube.shape[self.SPECTRAL_AXIS]
        self.row_size = self.data_cube.shape[self.ROW_AXIS] - self.pad_size * 2
        self.column_size = self.data_cube.shape[self.COLUMN_AXIS] - self.pad_size * 2

    def min_max_normalize(self):
        print("Min max normalization")
        normalized_data = np.empty_like(self.data_cube, dtype=float)
        for band_id in range(self.spectral_size):
            max_ = np.amax(self.data_cube[..., band_id])
            min_ = np.amin(self.data_cube[..., band_id])
            normalized_data[..., band_id] = (self.data_cube[..., band_id] - min_) / \
                                            (max_ - min_)
        self.data_cube = normalized_data

    @staticmethod
    def get_min(value: int) -> int:
        divisors = []
        for divisor in range(2, math.ceil(math.sqrt(value))):
            if value % divisor == 0:
                divisors.append(int(value / divisor))
        min_ = 1
        if len(divisors) != 0:
            min_ = np.min(divisors)
        return int(min_)

    @staticmethod
    def get_median(value):
        divisors = []
        for divisor in range(2, math.ceil(math.sqrt(value))):
            if value % divisor == 0:
                divisors.append(int(value / divisor))
        median = 1
        if len(divisors) != 0:
            if len(divisors) % 2 != 1:
                median = np.median(divisors)
            else:
                median = divisors[int(len(divisors) / 2)]
        return int(median)
