import argparse
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from utils import *
from dataset import ganzin_pupil_valid_data
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)
def valid(args, model, valid_loader):
    model.eval()
    pbar = tqdm(total=len(valid_loader), ncols=0, desc="Valid", unit=" step")

    total_loss = 0
    eps = 1e-7
    correct = [0, 0]
    total = [0, 0]
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for idx, (image, label) in enumerate(valid_loader):
            image = image.cuda()
            label = label.cuda()

            pred = model(image)
            pred = sigmoid(pred)
            pred = pred.squeeze()

            loss = criterion(pred, label)
            correct, total = get_valid_acc(pred, label, correct, total, args.threshold)

            total_loss += loss

            pbar.update()
            pbar.set_postfix(
                valid_open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                valid_close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                total_loss = f"{total_loss:.4f}"
            )
        valid_acc = (sum(correct) / sum(total)) * 100
        pbar.set_postfix(
            valid_open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
            valid_close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
            valid_acc = f"{valid_acc:.2f}%",
            total_loss = f"{total_loss:.4f}"
        )
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    parser.add_argument('--root', type=str, default="../dataset")
    parser.add_argument('--load', type=str, default='./checkpoint/ganzin/valid/model_best.pth')

    ''' paramters '''
    parser.add_argument('--img_size', type=int,default=[640, 480], help='size of image', nargs='+')
    parser.add_argument('--threshold', type=float, default=0.4, help='determine whether eyes are open')
    
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    Set_seed(args.seed)

    data_valid = np.loadtxt(os.path.join(args.root, 'valid_eye.txt'), delimiter=',', dtype=np.str)
    data_non_valid = np.loadtxt(os.path.join(args.root, 'non_valid_eye.txt'), delimiter=',', dtype=np.str)
    train_data_valid_list, valid_data_valid_list = train_test_split(data_valid, train_size=0.9)
    train_data_nonvalid_list, valid_data_nonvalid_list = train_test_split(data_non_valid, train_size=0.9)
    train_data_list = np.concatenate((train_data_valid_list, train_data_nonvalid_list), axis=0)
    valid_data_list = np.concatenate((valid_data_valid_list, valid_data_nonvalid_list), axis=0)

    valid_data = ganzin_pupil_valid_data(args, valid_data_list, isTrain=False)

    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers, drop_last=False, shuffle=False)
    print('Number of validation data : ', len(valid_data))
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.cuda()

    if args.load:
        print("Load pretrained model!!")
        model.load_state_dict(torch.load(args.load))

    
    valid(args, model, valid_loader)