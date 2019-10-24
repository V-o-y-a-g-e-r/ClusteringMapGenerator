import argparse
from typing import NamedTuple
import os
import torch

from python_research.experiments.unsupervised_segmentation.model import RecurrentAutoencoder
from python_research.experiments.unsupervised_segmentation.pipeline import Pipeline
from python_research.experiments.unsupervised_segmentation.spectral_dataset import SpectralDataset
from python_research.experiments.unsupervised_segmentation.utils import save_metrics


class Arguments(NamedTuple):
    data_path: str
    labels_path: str
    spatial: int
    dest_path: str
    n_clusters: int
    epochs: int
    patience: int


def arguments() -> Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--labels_path', dest='labels_path', type=str)
    parser.add_argument('--spatial', dest='spatial', type=int)
    parser.add_argument('--dest_path', dest='dest_path', type=str)
    parser.add_argument('--n_clusters', dest='n_clusters', type=int)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--patience', dest='patience', type=int)
    return Arguments(**vars(parser.parse_args()))


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args = arguments()
    os.makedirs(args.dest_path, exist_ok=True)
    dataset = SpectralDataset(args.spatial)
    dataset.load_data(args.data_path, args.labels_path)
    dataset.min_max_normalize()
    dataloader = dataset.get_dataloader()
    model = RecurrentAutoencoder(n_clusters=args.n_clusters, spatial_size=dataset.spatial_size,
                                 seq_len=dataset.spectral_size, img_height=dataset.row_size,
                                 img_width=dataset.column_size, gt=dataset.ground_truth,
                                 dest_path=args.dest_path).to(device)
    pipeline = Pipeline(model, dataloader, args.epochs, args.patience)
    pipeline.run_pipeline()
    save_metrics(args.dest_path, pipeline.metrics)


if __name__ == '__main__':
    main()
