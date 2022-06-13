from networks.deeplab.deeplab import DeepLab
from dataset import TEyeD_pupil_seg_data, ganzin_pupil_seg_data
from loss import DiceFocalLoss, DiceCeLoss, FocalLoss
from semseg.models.segformer import SegFormer
from semseg.models.sfnet import SFNet
from semseg.models.lawin import Lawin

from utils import *

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import torchvision

from tensorboardX import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('agg')
import cv2

import argparse
import os
import tqdm
from sklearn.model_selection import train_test_split
from monai.inferers import sliding_window_inference

def main(args, model, loss_func, optimizer, train_loader, valid_loader, scheduler):
    writer = SummaryWriter(log_dir='runs/%s' % args.model)
    model = model.cuda()
    loss_func = loss_func.cuda()

    max_score = 0
    training_step = 0
    
    for epoch in range(0, args.epochs):
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="train[%d/%d]" % (epoch, args.epochs), unit=" step")

        model.train()

        total_loss = 0.
        total, correct = 0, 0
        lr = optimizer.param_groups[0]['lr']

        iou_meter = AverageMeter()
        iou_meter_sequence = AverageMeter()
        label_validity = []
        output_conf = []
        for image, mask in train_loader:    
            image, mask = image.cuda(), mask.cuda()
            mask = mask[:,0,:,:].long()
            
            pred = model(image)

            optimizer.zero_grad()
            loss = loss_func(pred, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            for batch in range(pred_label.shape[0]):
                output = pred_label[batch]
                label = mask[batch]

                if np.sum(output.flatten()) > 0:
                    conf = 1.0
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    conf = 0.0

                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                    
                output_conf.append(conf)

            total, correct = get_positive_acc(pred_label, mask, total, correct)
            
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr:.5f}",
                acc=f"{(correct / total):.4f}"
            )
            writer.add_scalar('Training_loss', loss.item(), training_step)
            training_step += 1
        
        wiou = iou_meter.avg()
        writer.add_scalar('Training loss', total_loss, epoch)
        writer.add_scalar('Training wiou', wiou, epoch)
        if args.non_valid:
            tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
            atnr = np.mean(tn_rates)
            score = 0.7 * wiou + 0.3 * atnr
            writer.add_scalar('Training atnr', atnr, epoch)
            writer.add_scalar('Training score', score, epoch)

            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr:.5f}",
                acc=f"{(correct/total):.4f}",
                weighted_iou=f"{wiou:.4f}",
                negative_rate=f"{atnr:.4f}",
                Score=f"{score:.4f}"
            )
        else:
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                lr=f"{lr:.5f}",
                acc=f"{(correct/total):.4f}",
                weighted_iou=f"{wiou:.4f}",
            )

        pbar.close()
        if (epoch + 1) % args.val_freq == 0 or (epoch +1) == args.epochs:
            val_score = valid(args, model, loss_func, valid_loader, epoch, writer)
            if max_score < val_score:
                max_score = val_score
                torch.save(model.state_dict(), os.path.join(args.save_model, 'model_best.pth'))
                print("Save best model!!!")
            torch.save(model.state_dict(), os.path.join(args.save_model, 'model_last.pth'))
        scheduler.step()

def valid(args, model, loss_func, valid_loader, epoch, writer):
    model.eval()
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="val[%d/%d]" % (epoch, args.epochs), unit=" step")
        mean = torch.as_tensor([0.38314])
        std = torch.as_tensor([0.31113])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        count = 0
        total_loss = 0.
        total, correct = 0, 0

        iou_meter = AverageMeter()
        iou_meter_sequence = AverageMeter()
        label_validity = []
        output_conf = []
        for image, mask in valid_loader:
            image, mask = image.cuda(), mask.cuda()
            mask = mask[:,0,:,:].long()

            if args.model == 'lawin':
                pred = sliding_window_inference(image, (args.img_size[0], args.img_size[0]), 4, model, overlap=0.7)
            else:
                pred = model(image)
            
            loss = loss_func(pred, mask)
            total_loss += loss.item()
            pred = softmax(pred)

            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            total, correct = get_positive_acc(pred_label, mask, total, correct)

            for batch in range(pred_label.shape[0]):
                prob = pred[batch, 1, :]
                output = pred_label[batch]
                label = mask[batch]

                if np.sum(output.flatten()) > 0:
                    _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                    stats = stats[1:, :]
                    pupil_candidate = np.argmax(stats[:, 4]) + 1
                    output[labels != pupil_candidate] = 0
                    output = convex_hull(output.astype(np.uint8) * 255)
                    output = output / 255
                    pred_label[batch] = output

                    conf = np.mean(prob[output == 1.0])

                    if conf < args.conf_threshold:
                        conf = 0.0
                        pred_label[batch] = np.zeros(pred_label[batch].shape)
                    else:
                        conf = 1.0
                        iou = mask_iou(output, label)
                        iou_meter.update(conf * iou)
                        iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    conf = 0.0

                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                else:  # empty ground truth label
                    label_validity.append(0.0)

                output_conf.append(conf)

            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                acc=f"{(correct/total):.4f}"
            )

            """plot segmentation map"""
            image = image.cpu().detach()
            image = ((image * std) + mean)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
            plt.figure(figsize=(18,12))
            plt.subplot(1,3,1)
            plt.title("image")
            plt.imshow(image[0].transpose((1, 2, 0)))
            plt.subplot(1,3,2)
            plt.title("label")
            plt.imshow(mask[0])
            plt.subplot(1,3,3)
            plt.title("predcited")
            plt.imshow(pred_label[0])
            plt.savefig(os.path.join(args.save_fig, '%d.png' % count))
            plt.clf()
            plt.close()
            count += 1

        wiou = iou_meter.avg()
        writer.add_scalar('validation loss', total_loss, epoch)
        writer.add_scalar('validation wiou', wiou, epoch)
        if args.non_valid:
            tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
            atnr = np.mean(tn_rates)
            score = 0.7 * wiou + 0.3 * atnr
            writer.add_scalar('validation atnr', atnr, epoch)
            writer.add_scalar('validation score', score, epoch)

            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                acc=f"{(correct/total):.4f}",
                weighted_iou=f"{wiou:.4f}",
                negative_rate=f"{atnr:.4f}",
                Score=f"{score:.4f}"
            )
            pbar.close()
            return score
        else:
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                acc=f"{(correct/total):.4f}",
                weighted_iou=f"{wiou:.4f}",
            )
            pbar.close()
            return wiou
        
def get_args():
    parser = argparse.ArgumentParser(
        description='2022 cv final project -- Pupil tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Path
    parser.add_argument('--root', type=str, default='../dataset/', help='path to dataset')
    parser.add_argument('--train_txt', type=str, default='../dataset/train.txt', help='path to dataset')
    parser.add_argument('--valid_txt', type=str, default='../dataset/valid.txt', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='ganzin', choices=['TEyeD', 'ganzin'])
    parser.add_argument('--save_model', default='./checkpoint/TEyeD', help='path to save model')
    parser.add_argument('--save_fig', default='./out_mask', help='path to save predicted figures')
    parser.add_argument('--load', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('--non_valid', action='store_true', help='Whether to use non_valid eyes')

    # Model parameters
    parser.add_argument('--num_classes', type=int,default=2, help='Number of classes')
    parser.add_argument('--img_size', type=int,default=[384, 256], help='size of image', nargs='+')
    parser.add_argument('--model', type=str, default='deeplab', help='deeplab/segformer/sfnet/lawin')
    parser.add_argument('--conf_threshold', type=float, default=0.80, help='threshold for positive labels')

    # Hyperparameters
    parser.add_argument('--epochs', type=int,default=100, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--dice_weight', type=float, default=0.6, help='dice loss weight')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_freq', type=int, default=1, help='validation freq')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam optimizer(default=SGD)')

    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--scheduler', type=str, default='cosine', help='cosine/linearwarmup')
    parser.add_argument("--device", type=str, default='0,1', help="Select gpu")
    parser.add_argument('--seed', type=int,default=2022, help='seed')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    Set_seed(args.seed)

    print("Used model: ", args.model)
    print("non_valid training", args.non_valid)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.save_model = os.path.join(args.save_model, args.model)
    args.save_fig = args.save_fig + '_' + args.model
    os.makedirs(args.save_model, exist_ok=True)
    os.makedirs(args.save_fig, exist_ok= True)

    if args.dataset == 'TEyeD':
        data_list = os.listdir(os.path.join(args.root, args.dataset, 'images'))
        train_data_list, valid_data_list = train_test_split(data_list, train_size=0.9)
        train_data = TEyeD_pupil_seg_data(args, train_data_list, isTrain=True)
        valid_data = TEyeD_pupil_seg_data(args, valid_data_list, isTrain=False)
    elif args.dataset == 'ganzin':
        train_data = ganzin_pupil_seg_data(args, isTrain=True, nonvalid=args.non_valid)
        valid_data = ganzin_pupil_seg_data(args, isTrain=False, nonvalid=args.non_valid)
    
    print('Number of training data: ', len(train_data))
    print('Number of validation data: ', len(valid_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size // 2, num_workers=args.workers, shuffle=False, pin_memory=True)

    if args.model == 'deeplab':
        model = DeepLab(backbone='resnet')
        model.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
    elif args.model == 'segformer':
        model = SegFormer(backbone='MiT-B2', num_classes=150)
        model.load_state_dict(torch.load('./pretrained/segformer.b2.ade.pth', map_location='cpu'))
        model.decode_head.linear_pred = nn.Conv2d(768, args.num_classes, kernel_size=(1,1), stride=(1,1))
    elif args.model == 'sfnet':
        model = SFNet('ResNetD-18')
        model.load_state_dict(torch.load('./pretrained/SFNet_18_HELEN_61_0.pth', map_location='cpu'))
        model.head.conv_seg = nn.Conv2d(128, args.num_classes, 1)
    elif args.model == 'lawin':
        model = Lawin('MiT-B2')
        model.decode_head.linear_pred = nn.Conv2d(512, args.num_classes, 1)

    if args.model == 'segformer' or args.model == 'lawin':
        model = nn.DataParallel(model)

    if args.load:
        print('Load pretrained model!!')
        model.load_state_dict(torch.load(args.load))

    if args.adam:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)

    loss_func = DiceFocalLoss(args)
    # loss_func = DiceCeLoss(args)

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'linearwarmup':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.epochs)


    main(args, model, loss_func, optimizer, train_loader, valid_loader, scheduler)


