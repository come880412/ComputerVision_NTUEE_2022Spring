import argparse
from statistics import mode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import tqdm
import os

from utils import get_acc, fixed_seed
from dataset import get_cifar10_train_val_set
from myModel import *

import warnings
warnings.filterwarnings("ignore")

def train_batch(model, optimizer, criterion, image, label):
    optimizer.zero_grad()
    pred = model(image)
    loss = criterion(pred, label)

    loss.backward()
    # if the gradient is too large, we dont adopt it
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)

    optimizer.step()

    return loss, pred

def validation(opt, model, criterion, val_loader, writer, epoch):
    model.eval()

    total_correct = 0
    total_label = 0
    val_loss = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label in val_loader:
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()

            pred = model(image)
            loss = criterion(pred, label)

            correct, total = get_acc(pred, label)

            total_label += total
            total_correct += correct
            val_acc = (total_correct / total_label) * 100

            val_loss += loss

            label = label.cpu().detach()
            pred = pred.cpu().detach()

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                Accuracy=f"{val_acc:.2f}"
            )
    
    pbar.set_postfix(
        loss=f"{val_loss:.4f}",
        Accuracy=f"{val_acc:.2f}",
    )
    pbar.close()
    
    writer.add_scalar('validation loss', val_loss / len(val_loader.dataset), epoch)
    writer.add_scalar('validation acc', val_acc, epoch)

    return val_acc, val_loss

def main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % opt.model)
    
    criterion = criterion.cuda()
    model = model.cuda()

    """training"""
    print('Start training!')
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    max_acc = 0.

    for epoch in range(0, opt.n_epochs):
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")

        total_loss = 0
        total_correct = 0
        total_label = 0

        for image, label in train_loader:
            image, label = image.cuda(), label.cuda()
            train_loss, pred = train_batch(model, optimizer, criterion, image, label)

            correct, total = get_acc(pred, label)

            total_label += total
            total_correct += correct
            acc = (total_correct / total_label) * 100

            total_loss += train_loss
        
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                Accuracy=f"{acc:.2f}%"
            )

        train_loss = total_loss / len(train_loader.dataset)

        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('training acc', acc, epoch)

        pbar.close()

        val_acc, val_loss = validation(opt, model, criterion, val_loader, writer, epoch)
        if max_acc <= val_acc:
            print('save model!!')
            max_acc = val_acc                
            torch.save(model.state_dict(), os.path.join(opt.save_model, opt.model, 'model_epoch%d_acc%.2f.pth' % (epoch, max_acc)))
            torch.save(model.state_dict(), os.path.join(opt.save_model, opt.model, 'model_best.pth'))

        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        scheduler.step(val_loss)

    print('best ACC:%.2f' % (max_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--lr_decay_rate", type=int, default=0.1, help="Start to decay epoch")

    parser.add_argument('--root', default='./p2_data/annotations/train_annos.json', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

    parser.add_argument('--optimizer', default='sgd', help='adam/sgd')
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")

    parser.add_argument('--model', default='myResnet', help='Lenet/myResnet')

    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu workers')
    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--save_model', default='./checkpoints', help='path to save model')

    parser.add_argument('--clean_data', type=int, default=1, help='Whether to clean the dataset')
    parser.add_argument('--weak_model_load', default='./checkpoints/weak_model/myResnet/model_epoch7_acc63.70.pth', help='path to weak model')
    parser.add_argument('--clean_threshold', type=float, default=0.23, help='threshold for keeping the data')

    parser.add_argument('--semi', type=int, default=1, help='Whether to do semi-supervised learning')
    parser.add_argument('--semi_model_load', default='./checkpoints/semi_model/myResnet/model_best.pth', help='path to weak model')
    parser.add_argument('--semi_threshold', type=float, default=0.9, help='threshold for adding the data')
    
    opt = parser.parse_args()
    fixed_seed(2022)
    os.makedirs(os.path.join(opt.save_model, opt.model), exist_ok=True)        

    train_data, valid_data = get_cifar10_train_val_set(opt, opt.root, ratio=0.9)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
    val_loader = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    if opt.model == 'Lenet':
        model = myLeNet(opt.num_classes)
    elif opt.model == 'myResnet':
        model = myResnet(num_out=opt.num_classes)
    
    if opt.load != '':
        print(f'loading pretrained model from {opt.load}')
        model.load_state_dict(torch.load(opt.load))

    criterion = torch.nn.CrossEntropyLoss()

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay = 7e-5)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay = 7e-5, momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay_rate, patience=2, min_lr=1e-5)

    main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader)