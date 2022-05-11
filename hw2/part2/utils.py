import numpy as np
import os

import cv2

import torch
import random

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def get_acc(y_pred, y_true):
    """ ACC metric
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    task: the task of the current dataset(multi-label or multi-class)
    threshold: the threshold for multilabel
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    y_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(np.equal(y_true, y_pred))
    total = y_true.shape[0]
    
    return correct, total

def Norm():
    img_h, img_w = 32, 32
    # img_h, img_w = 224, 224   #根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    
    train_path = './p2_data/train'
    image_names = os.listdir(train_path)
    imgs_path_list = []
    for image_name in image_names:
        imgs_path_list.append(os.path.join(train_path, image_name))
    
    len_ = len(imgs_path_list)
    i = 0
    for path in imgs_path_list:
        img = cv2.imread(path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i%1000 == 0:
            print(i,'/',len_)    
    
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
