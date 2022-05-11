import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import tqdm
import json
import os

from utils import get_acc
from myModel import *
from dataset import Cifar10_data

import warnings
warnings.filterwarnings("ignore")

def test(model, test_loader):
    model.eval()

    cnt = 0
    for image, label in tqdm.tqdm(test_loader):
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()

            pred = model(image)

            pred = torch.argmax(pred, axis=1)
            cnt += (pred.eq(label.view_as(pred)).sum().item())

    acc = cnt / len(test_loader.dataset)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_anno', help='annotaion for test image', type=str, default= './p2_data/annotations/public_test_annos.json')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--path', default='./checkpoints/myResnet/model_best.pth', help='path to model to continue training')
    
    opt = parser.parse_args()

    model = myResnet(num_out=opt.num_classes)

    model = model.cuda()
    
    if opt.path != '':
        print(f'loading pretrained model from {opt.path}')
        model.load_state_dict(torch.load(opt.path))


    with open(opt.test_anno, 'r') as f :
        data = json.load(f)    
    
    imgs, categories = data['images'], data['categories']
    
    means = [0.500, 0.478, 0.433]
    stds = [0.248, 0.246, 0.264]
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])
    
    test_set = Cifar10_data(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/public_test/')
    #test_set = Cifar10_data(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/private_test/')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    acc = test(model, test_loader)
    print("accuracy : ", acc)