import os

import numpy as np
import xlsxwriter
from skimage import img_as_ubyte
from skimage.color import label2rgb
from skimage.io import imsave
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from python_research.experiments.unsupervised_segmentation.pipeline import MetricsEnum


def save_generated_map(gt_hat: np.ndarray, dest_path: str, epoch: int):
    imsave(os.path.join(dest_path, '{}-epoch-generated-map.png'.format(epoch)),
           img_as_ubyte(label2rgb(gt_hat)))
    np.save(os.path.join(dest_path, '{}-epoch-generated-map.npy'.format(epoch)), gt_hat.astype(np.uint8))


def save_gt_based_map(gt_hat: np.ndarray, gt: np.ndarray, dest_path: str, epoch: int):
    background = np.where(gt == 0)
    for row, column in zip(*background):
        gt_hat[row, column] = -1
    imsave(os.path.join(dest_path, '{}-epoch-gt-based-map.png'.format(epoch)),
           img_as_ubyte(label2rgb(gt_hat)))


def save_metrics(dest_path: str, artifacts: dict):
    workbook = xlsxwriter.Workbook(os.path.join(dest_path, 'metrics.xlsx'))
    worksheet = workbook.add_worksheet()
    row, col = 0, 0
    for key in artifacts.keys():
        row = 1
        col += 1
        worksheet.write(row, col, key.value)
        for item in artifacts[key]:
            row += 1
            worksheet.write(row, col, item)
    workbook.close()


def get_metrics(gt_hat: np.ndarray, gt: np.ndarray, metrics: dict):
    gt_hat = gt_hat.ravel()
    gt = gt.ravel()
    to_delete = np.where(gt == 0)[0]
    gt, gt_hat = np.delete(gt, to_delete), np.delete(gt_hat, to_delete)
    ars_score = adjusted_rand_score(gt, gt_hat)
    nmi_score = normalized_mutual_info_score(gt, gt_hat)
    metrics[MetricsEnum.ARS_SCORE].append(ars_score)
    metrics[MetricsEnum.NMI_SCORE].append(nmi_score)
    print("ARS -> {}\nNMI -> {}".format(ars_score, nmi_score))
