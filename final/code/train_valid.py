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
def valid(args, model, valid_loader, criterion, epoch):
    model.eval()
    pbar = tqdm(total=len(valid_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, args.epochs), unit=" step")

    total_loss = 0
    eps = 1e-7
    correct = [0, 0]
    total = [0, 0]
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for idx, (image, label) in enumerate(valid_loader):
            image = image.cuda()
            label = label.cuda()

            pred = model(image)
            pred = pred.squeeze()

            loss = criterion(pred, label)
            pred = sigmoid(pred)
            correct, total = get_valid_acc(pred, label, correct, total, args.threshold)

            total_loss += loss

            pbar.update()
            pbar.set_postfix(
                open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                total_loss = f"{total_loss:.4f}"
            )
        valid_acc = (sum(correct) / sum(total)) * 100
        open_acc = (correct[1]/(total[1] + eps)) * 100
        close_acc = (correct[0]/(total[0] + eps))*100

        pbar.set_postfix(
                open_acc = f"{open_acc:.2f}%",
                close_acc = f"{close_acc:.2f}%",
                acc = f"{valid_acc:.2f}%",
                total_loss = f"{total_loss:.4f}"
            )
        pbar.close()
        return total_loss, valid_acc

def train(image, label, model, optimizer, criterion):
    sigmoid = nn.Sigmoid()

    optimizer.zero_grad()
    pred = model(image)
    pred = pred.squeeze()

    loss = criterion(pred, label)

    loss.backward()
    optimizer.step()

    return sigmoid(pred), loss

def main(args, model, train_loader, valid_loader, optimizer, scheduler, criterion):

    print('start tarining...')

    max_acc = 0.
    eps = 1e-7
    
    for epoch in range(args.epochs):
        pbar = tqdm(total=len(train_loader), ncols=0, desc="train[%s/%s]" % (epoch, args.epochs), unit=" step")

        model.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        correct = [0, 0]
        total = [0, 0]
        for image, label in train_loader:
            label = label.squeeze()
            image, label = image.cuda(), label.cuda()

            pred, train_loss = train(image, label, model, optimizer, criterion)
            correct, total = get_valid_acc(pred, label, correct, total, args.threshold)

            pbar.update()
            pbar.set_postfix(
                train_loss = f"{train_loss:.4f}",
                open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                lr=f"{lr:.6f}"
            )

        train_acc = (sum(correct) / sum(total)) * 100
        pbar.set_postfix(
                open_acc = f"{(correct[1]/(total[1] + eps))*100:.2f}%",
                close_acc = f"{(correct[0]/(total[0] + eps))*100:.2f}%",
                acc = f"{train_acc:.2f}%",
                train_loss = f"{train_loss:.4f}"
            )
        pbar.close()
        valid_loss, valid_acc = valid(args, model, valid_loader, criterion, epoch)

        if valid_acc >= max_acc:
            print("Save model!!")
            max_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.saved_model, 'model_best.pth'))
        torch.save(model.state_dict(), os.path.join(args.saved_model, 'model_last.pth'))

        scheduler.step()
    print('max_acc: ', max_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' Paths '''
    parser.add_argument('--root', type=str, default="../dataset", help='neurobit_valid/TEyeD_valid')
    parser.add_argument('--saved_model', type=str, default='./checkpoint/ganzin/valid')
    parser.add_argument('--load', type=str, default='./checkpoint/TEyeD/valid/model_best.pth')
    parser.add_argument('--model', type=str, default='resnet18', help='resnet18')

    ''' paramters '''
    parser.add_argument('--img_size', type=int,default=[640, 480], help='size of image', nargs='+')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--warmup_epochs', default=25, type=int, help='number of warmup epochs')
    parser.add_argument('--threshold', type=float, default=0.4, help='determine whether eyes are open')
    parser.add_argument('--scheduler', type=str, default='linearwarmup', help='cosine/linearwarmup')
    
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    os.makedirs(args.saved_model, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    Set_seed(args.seed)

    data_valid = np.loadtxt(os.path.join(args.root, 'valid_eye.txt'), delimiter=',', dtype=np.str)
    data_non_valid = np.loadtxt(os.path.join(args.root, 'non_valid_eye.txt'), delimiter=',', dtype=np.str)
    train_data_valid_list, valid_data_valid_list = train_test_split(data_valid, train_size=0.9)
    train_data_nonvalid_list, valid_data_nonvalid_list = train_test_split(data_non_valid, train_size=0.9)

    len_valid, len_nonvalid = len(train_data_valid_list), len(train_data_nonvalid_list)
    train_data_list = np.concatenate((train_data_valid_list, train_data_nonvalid_list), axis=0)
    valid_data_list = np.concatenate((valid_data_valid_list, valid_data_nonvalid_list), axis=0)

    train_data = ganzin_pupil_valid_data(args, train_data_list)
    valid_data = ganzin_pupil_valid_data(args, valid_data_list, isTrain=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers, drop_last=False, shuffle=False)
    print('Number of training data : ', len(train_data))
    print('Number of validation data : ', len(valid_data))
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.cuda()

    if args.load:
        print("Load pretrained model!!")
        model.load_state_dict(torch.load(args.load))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'linearwarmup':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.epochs)
    
    valid_weight = (len_nonvalid / len_valid) * 6
    class_weight = torch.FloatTensor([valid_weight])

    criterion = nn.BCEWithLogitsLoss(pos_weight = class_weight).cuda()

    main(args, model, train_loader, valid_loader, optimizer, scheduler, criterion)