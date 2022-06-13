import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
import math
import warnings
from typing import List
import os
import cv2
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch import nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def dice(pr, gt, eps=1e-7):
    """Calculate Dice cofficient between ground truth and prediction
    Args:
        pr (numpy.array): predicted map
        gt (numpy.array):  ground truth map
        eps (float): epsilon to avoid zero division
    Returns:
        float: dice score
    """
    TP, P, PP = 0, 0, 0
    for i in range(len(pr)):
        TP += np.sum(pr[i] * gt[i])
        P += np.sum(gt[i])
        PP += np.sum(pr[i])
        # Recall = TP / (P + eps)
        # Precision = TP / (PP + eps)
        # dice_score += (2 * Precision * Recall) / (Precision + Recall + eps)

    return TP, P, PP

def get_positive_acc(pr, gt, total, correct):
    correct += np.sum(pr * gt)
    total += np.sum(gt)

    return total, correct

def get_valid_acc(y_pred, y_true, correct, total, threshold=0.9):
        """ ACC metric
        y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
        y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        for i in range(y_true.shape[0]):
            if y_pred[i] >= threshold:
                pred = 1
            else:
                pred = 0
            label = y_true[i]
            total[int(label)] += 1

            if pred == label:
                correct[int(pred)] += 1
        
        return correct, total

def convex_hull(y_pred):
    imageHeight, imageWidth = y_pred.shape
    _, binary = cv2.threshold(y_pred, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_mask = np.zeros((imageHeight, imageWidth))
    for contour in contours:
        hull = cv2.convexHull(contour, False)
        pred_mask = cv2.drawContours(pred_mask, hull[np.newaxis,:,:], -1, 255, cv2.FILLED).astype('uint8')
    return pred_mask

def isvalidarea(src, th_area = 1000):
    mask = src.copy()
    
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    area = ([cv2.contourArea(contour) for contour in contours])[0]

    if area >= th_area:
        return True
    else:
        return False

def pupilTrack(src: np.ndarray, threshold: int = 180, ratio: float = 0.8) -> np.ndarray:
    """
        This program is used to detect the pupil part and output the detection result.
        :param src: np.ndarray -> input image.
        :param threshold: int -> threshold of binary method.
        :param ratio: float -> The ratio of the mask to the ellipse, if the ratio reaches a certain value, the mask will be filled.
        :return: output: np.ndarray - > output image with mask.
    """
    img = src.copy()

    # Step1. use GaussianBlur to reduce noise.
    img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=2)

    # Step2. Gamma Transform
    gamma = 0.1
    img_gamma = np.power(img.copy() / 255.0, gamma)
    img_gamma = img_gamma * 255.0
    img_gamma = img_gamma.astype(np.uint8)

    img_threshold = np.zeros(shape=img.shape, dtype=np.uint8)

    # img_threshold[np.where(img_gamma > int(np.mean(img_gamma) - np.std(img_gamma)))] = 255
    img_threshold[np.where(img_gamma > threshold)] = 255

    plt.imshow(img_threshold, cmap='gray')
    plt.show()

    # Step3. Find Contours
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Step4. Check each area of the contours
    area = ([cv2.contourArea(contour) for contour in contours])

    idx = -1
    for i in np.where(area > np.array(1000))[0]:
        if area[i] != np.max(area):
            idx = i

    output = np.zeros(shape=img.shape, dtype=np.uint8)

    if idx == -1:
        return np.zeros(shape=img.shape, dtype=np.uint8)

    # Step5. Add a mask to the pupil.
    ellipse = cv2.fitEllipse(contours[idx])
    outputArea = np.pi * ellipse[1][0] * ellipse[1][1] / 4

    return cv2.ellipse(output, ellipse, 255, -1) if area[idx] / outputArea >= ratio else output

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count

def true_negative_curve(confs: np.ndarray, labels: np.ndarray, nr_thresholds: int = 1000):
    """Compute true negative rates
    Args:
        confs: the algorithm outputs
        labels: the ground truth labels
        nr_thresholds: number of splits for sliding thresholds

    Returns:

    """
    thresholds = np.linspace(0, 1, nr_thresholds)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (confs < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - labels) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - labels))
    return np.array(tn_rates)

def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    """Calculate the IoU score between two segmentation masks
    Args:
        mask1: 1st segmentation mask
        mask2: 2nd segmentation mask
    """
    if len(mask1.shape) == 3:
        mask1 = mask1.sum(axis=-1)
    if len(mask2.shape) == 3:
        mask2 = mask2.sum(axis=-1)
    area1 = cv2.countNonZero((mask1 > 0).astype(int))
    area2 = cv2.countNonZero((mask2 > 0).astype(int))
    if area1 == 0 or area2 == 0:
        return 0
    area_union = cv2.countNonZero(((mask1 + mask2) > 0).astype(int))
    area_inter = area1 + area2 - area_union
    return area_inter / area_union

def benchmark(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            iou_meter_sequence.reset()
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(image_folder, f'{idx}.png')
                image = cv2.imread(image_name)
                label = cv2.imread(label_name)
                # TODO: Modify the code below to run your method or load your results from disk
                # output, conf = my_awesome_algorithm(image)
                output = label
                conf = 1.0
                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                output_conf.append(conf)
            # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Benchmark score: {score:.4f}')

    return score

def Set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class LinearWarmupCosineAnnealingLR(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended

if __name__ == '__main__':
    # dataset_path = r'D:\CV22S_Ganzin_final_project\dataset\public\S1\01'
    subject = [str(num).zfill(2) for num in range(1, 27)]
    for s in subject:
        dataset_path = '../dataset/public/S5/%s' % s
        mask_path = './public_mask_deeplab1_withcvmethod_dialted/S5_solution/%s' % s
        nr_image = len([name for name in os.listdir(dataset_path) if name.endswith('.jpg')])
        print(nr_image)
        image = cv2.imread(os.path.join(dataset_path, '0.jpg'))
        h = image.shape[0]
        w = image.shape[1]
        dpi = matplotlib.rcParams['figure.dpi']
        fig = plt.figure(figsize=(w / dpi, h / dpi))
        ax = fig.add_axes([0, 0, 1, 1])
        for idx in range(nr_image):
            image_name = os.path.join(dataset_path, f'{idx}.jpg')
            label_name = os.path.join(mask_path, f'{idx}.png')
            image = cv2.imread(image_name)
            label = cv2.imread(label_name)
            # print(np.max(label[:, :, 0]), np.max(label[:, :, 1]), np.max(label[:, :, 2]))
            blended = alpha_blend(image, label, 0.5)
            ax.clear()
            ax.imshow(blended)
            ax.axis('off')
            plt.draw()
            plt.pause(0.0005)
        plt.close()