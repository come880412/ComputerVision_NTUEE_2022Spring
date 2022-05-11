from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch

import os
from myModel import *

import numpy as np
from PIL import Image
import json
import tqdm


transforms_test =   transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.500, 0.478, 0.433), (0.248, 0.246, 0.264)),])

transforms_train =  transforms.Compose([
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.500, 0.478, 0.433), (0.248, 0.246, 0.264)),
                    ])

def clean_data(opt, imgs, categories):
    print("Start cleaning data")
    weak_model = myResnet(num_out=opt.num_classes)
    if opt.weak_model_load != '':
        print(f'loading pretrained model from {opt.weak_model_load}')
        weak_model.load_state_dict(torch.load(opt.weak_model_load))
    weak_model = weak_model.cuda()
    
    train_set = Cifar10_data(images=imgs, labels= categories, transform=transforms_test, prefix = './p2_data/train/')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)

    weak_model.eval()
    softmax = nn.Softmax()

    images = []
    labels = []
    count = 0
    idx = 0

    for image, label in tqdm.tqdm(train_loader):
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()

            pred = weak_model(image)
            pred = softmax(pred)
            pred = pred.cpu().detach().numpy()
            pred_max_prob = np.max(pred, axis=1)

            for prob in pred_max_prob:
                if prob >= opt.clean_threshold:
                    count += 1
                    images.append(imgs[idx])
                    labels.append(categories[idx])
                idx += 1

    print("After clearning the dataset")
    print("Number of images is ", len(images))
    print()
    return images, labels

def semi(opt):
    print("Start doing semi-supervised")
    model = myResnet(num_out=opt.num_classes)
    if opt.semi_model_load != '':
        print(f'loading pretrained model from {opt.semi_model_load}')
        model.load_state_dict(torch.load(opt.semi_model_load))
    model = model.cuda()
    
    imgs = os.listdir('./p2_data/unlabeled')
    imgs.sort()
    unlabel_data = Cifar10_data(images=imgs, labels= None, transform=transforms_test, prefix = './p2_data/unlabeled/')
    unlabel_loader = DataLoader(unlabel_data, batch_size=64, shuffle=False)

    model.eval()
    softmax = nn.Softmax()

    semi_images = []
    semi_labels = []

    idx = 0
    for image in tqdm.tqdm(unlabel_loader):
        with torch.no_grad():
            image = image.cuda()

            pred = model(image)
            pred = softmax(pred)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            pred_max_prob = np.max(pred, axis=1)

            for i, prob in enumerate(pred_max_prob):
                if prob >= opt.semi_threshold:
                    semi_images.append(imgs[idx])
                    semi_labels.append(pred_label[i])
                idx += 1

    print("Number of semi-images is ", len(semi_images))
    print()
    return semi_images, semi_labels

def get_cifar10_train_val_set(opt, root, ratio=0.9):
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    if opt.clean_data:
        images, labels = clean_data(opt, images, labels)
    if opt.semi:
        semi_images, semi_labels = semi(opt)

    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]

    if opt.semi:
        train_image = train_image + semi_images
        train_label = train_label + semi_labels

    train_set, val_set = Cifar10_data(images=train_image, labels=train_label,transform=transforms_train), \
                         Cifar10_data(images=val_image, labels=val_label,transform=transforms_test)
    
    return train_set, val_set

class Cifar10_data(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        try:
            image_path = os.path.join(self.prefix, self.images[index])
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = os.path.join('./p2_data/unlabeled', self.images[index])
            image = Image.open(image_path).convert("RGB")
        
        if self.labels:
            label = self.labels[index]
            return self.transform(image), label
        else:
            return self.transform(image)
