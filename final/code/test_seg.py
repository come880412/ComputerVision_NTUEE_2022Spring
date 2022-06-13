from networks.deeplab.deeplab import DeepLab
from dataset import ganzin_pupil_seg_data, ganzin_pupil_seg_public_data
from loss import DiceFocalLoss

from semseg.models.segformer import SegFormer
from semseg.models.sfnet import SFNet
from semseg.models.lawin import Lawin
from monai.inferers import sliding_window_inference

from utils import *

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('agg')

import argparse
import os
import tqdm
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

def valid(args, model_seg, model_valid, loss_func):
    args.save_fig = args.save_fig + '_' + args.model
    os.makedirs(args.save_fig, exist_ok= True)
    model_seg.eval()
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    valid_data = ganzin_pupil_seg_data(args, isTrain=False, nonvalid=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size // 2, num_workers=args.workers, shuffle=True, pin_memory=True)
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(valid_loader), ncols=0, desc="val", unit=" step")
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

            valid_preds = sigmoid(model_valid(image))
            valid_preds = valid_preds.cpu().detach().numpy()

            if args.model == 'lawin':
                pred = sliding_window_inference(image, (args.img_size[0], args.img_size[0]), 4, model_seg, overlap=0.7)
            else:
                pred = model_seg(image)
            
            loss = loss_func(pred, mask)
            total_loss += loss.item()
            pred = softmax(pred)

            mask = mask.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            image = image.cpu().detach()
            image = ((image * std) + mean)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
            for batch in range(pred_label.shape[0]):
                prob = pred[batch, 1, :]
                output = pred_label[batch].astype(np.uint8)
                label = mask[batch].astype(np.uint8)
                valid_pred = valid_preds[batch]
                
                if args.valid:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255) / 255

                        conf = np.mean(prob[output == 255.0])
                        # kernel = np.ones((3, 3), np.uint8)
                        # output = cv2.dilate(output, kernel=kernel, iterations=1)

                        # if not isvalidarea(output, th_area=1000):
                        #     image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        #     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        #     output = pupilTrack(image_cv)

                        pred_label[batch] = output
                        
                        if conf < args.conf_threshold or valid_pred < args.val_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                            iou = mask_iou(output, label)
                            iou_meter.update(conf * iou)
                            iou_meter_sequence.update(conf * iou)

                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0

                else:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255) / 255

                        conf = np.mean(prob[output == 255.0])
                        # kernel = np.ones((3, 3), np.uint8)
                        # output = cv2.dilate(output, kernel=kernel, iterations=1)

                        # if not isvalidarea(output, th_area=1000):
                        #     image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        #     image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        #     output = pupilTrack(image_cv)

                        pred_label[batch] = output
                        
                        if conf < args.conf_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                            iou = mask_iou(output, label)
                            iou_meter.update(conf * iou)
                            iou_meter_sequence.update(conf * iou)

                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0

                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                else:  # empty ground truth label
                    label_validity.append(0.0)

                output_conf.append(conf)

                """plot segmentation map"""
                # output = output.astype(np.uint8)
                # label = label.astype(np.uint8)
                # plt.figure(figsize=(18,12))
                # plt.subplot(1,3,1)
                # plt.title("image")
                # plt.imshow(image[batch].transpose((1, 2, 0)))
                # plt.subplot(1,3,2)
                # plt.title("label")
                # plt.imshow(label)
                # plt.subplot(1,3,3)
                # plt.title("predcited")
                # plt.imshow(output)
                # plt.savefig(os.path.join(args.save_fig, '%d.png' % count))
                # plt.clf()
                # plt.close()
                # count += 1

            total, correct = get_positive_acc(pred_label, mask, total, correct)
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
            )
        tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
        wiou = iou_meter.avg()
        atnr = np.mean(tn_rates)
        score = 0.7 * wiou + 0.3 * atnr

        pbar.set_postfix(
            loss=f"{total_loss:.4f}",
            acc=f"{(correct/total):.4f}",
            weighted_iou=f"{wiou:.4f}",
            negative_rate=f"{atnr:.4f}",
            Score=f"{score:.4f}"
        )
        pbar.close()

def test(args, model_seg, model_valid):
    conf_dict = {}
    softmax = nn.Softmax(dim=1)
    for id in range(1, 27):
        os.makedirs(os.path.join(args.public_mask, args.subject + '_solution', str(id).zfill(2)), exist_ok=True)
        conf_dict[str(id).zfill(2)] = []

    # os.makedirs(args.public_mask +'_visual', exist_ok=True)
    public_data = ganzin_pupil_seg_public_data(data_path=args.root, subject=args.subject)

    print('Number of public data: ', len(public_data))
    public_loader = DataLoader(public_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(public_loader), ncols=0, desc="public", unit=" step")
        mean = torch.as_tensor([0.38314])
        std = torch.as_tensor([0.31113])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        for image, data_info in public_loader:
            subject, id, image_name = data_info
            image = image.cuda()
            
            valid_preds = sigmoid(model_valid(image))
            valid_preds = valid_preds.cpu().detach().numpy()

            if args.model == 'lawin':
                pred = sliding_window_inference(image, (args.img_size[0], args.img_size[0]), 4, model_seg, overlap=0.7)
            else:
                pred = model_seg(image)
            
            pred = softmax(pred)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            image = image.cpu().detach()
            image = ((image * std) + mean)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
            for batch in range(pred_label.shape[0]):
                prob = pred[batch, 1, :]
                output = pred_label[batch]
                valid_pred = valid_preds[batch]

                if args.valid:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255)

                        conf = np.mean(prob[output == 255.0])
                        kernel = np.ones((3, 3), np.uint8)
                        output = cv2.dilate(output, kernel=kernel, iterations=1)

                        if not isvalidarea(output, th_area=1000):
                            image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                            output = pupilTrack(image_cv)

                        pred_label[batch] = output / 255
                        
                        if conf < args.conf_threshold or valid_pred < args.val_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0

                else:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255)

                        conf = np.mean(prob[output == 255.0])
                        kernel = np.ones((3, 3), np.uint8)
                        output = cv2.dilate(output, kernel=kernel, iterations=1)

                        if not isvalidarea(output, th_area=1000):
                            image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                            output = pupilTrack(image_cv)

                        pred_label[batch] = output / 255
                        
                        if conf < args.conf_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0
                
                conf_dict[id[batch]].append(conf)
                """plot segmentation map"""
                pred_label[batch] = pred_label[batch].astype(np.uint8)
                # if conf == 0.0:
                # if conf > 0.3 and conf <=0.9:
                # if valid_pred < args.val_threshold and np.sum(output.flatten()) > 0:
                # print(valid_pred)
                # plt.figure(figsize=(18,12))
                # plt.subplot(1,2,1)
                # plt.title("image")
                # plt.imshow(image[batch].transpose((1, 2, 0)))
                # plt.subplot(1,2,2)
                # plt.title("predcited")
                # plt.imshow(pred_label[batch])
                # plt.savefig(os.path.join(args.public_mask + '_visual', '%s.png' % image_name[batch][:-4]))
                # plt.clf()
                # plt.close()

                out_mask = np.stack((pred_label[batch], pred_label[batch], pred_label[batch]), axis=2)
                out_mask[:, :, 1] = 0
                cv2.imwrite(os.path.join(args.public_mask, subject[batch] + '_solution', id[batch], image_name[batch][:-4] + '.png'), out_mask * 255)
                
            pbar.update()

        pbar.close()
        for id in conf_dict.keys():
            save_path = os.path.join(args.public_mask, args.subject + '_solution', id, 'conf.txt')
            np.savetxt(save_path,  conf_dict[id], fmt='%.1f', delimiter=',')

def test_challenge(args, model_seg, model_valid):
    conf_list = []
    softmax = nn.Softmax(dim=1)
    folder_name = args.root.split('/')[-1]
    os.makedirs(os.path.join(args.public_mask, folder_name + '_solution'), exist_ok=True)

    public_data = ganzin_pupil_seg_public_data(data_path=args.root, subject=args.subject)

    print('Number of public data: ', len(public_data))
    public_loader = DataLoader(public_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    sigmoid = nn.Sigmoid()
    
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(public_loader), ncols=0, desc="public", unit=" step")
        mean = torch.as_tensor([0.38314])
        std = torch.as_tensor([0.31113])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        for image, data_info in public_loader:
            image_name = data_info
            image = image.cuda()
            
            valid_preds = sigmoid(model_valid(image))
            valid_preds = valid_preds.cpu().detach().numpy()

            if args.model == 'lawin':
                pred = sliding_window_inference(image, (args.img_size[0], args.img_size[0]), 4, model_seg, overlap=0.7)
            else:
                pred = model_seg(image)
            
            pred = softmax(pred)
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            image = image.cpu().detach()
            image = ((image * std) + mean)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
            for batch in range(pred_label.shape[0]):
                prob = pred[batch, 1, :]
                output = pred_label[batch]
                valid_pred = valid_preds[batch]

                if args.valid:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255)

                        conf = np.mean(prob[output == 255.0])
                        kernel = np.ones((3, 3), np.uint8)
                        output = cv2.dilate(output, kernel=kernel, iterations=1)

                        if not isvalidarea(output, th_area=1000):
                            image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                            output = pupilTrack(image_cv)

                        pred_label[batch] = output / 255
                        
                        if conf < args.conf_threshold or valid_pred < args.val_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0

                else:
                    if np.sum(output.flatten()) > 0:
                        # Remove other segmentation result
                        _, labels, stats, center = cv2.connectedComponentsWithStats(output[:, :].astype(np.uint8)) # 0: background
                        stats = stats[1:, :]
                        pupil_candidate = np.argmax(stats[:, 4]) + 1
                        output[labels != pupil_candidate] = 0
                        output = convex_hull(output.astype(np.uint8) * 255)

                        conf = np.mean(prob[output == 255.0])
                        kernel = np.ones((3, 3), np.uint8)
                        output = cv2.dilate(output, kernel=kernel, iterations=1)

                        if not isvalidarea(output, th_area=1000):
                            image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                            output = pupilTrack(image_cv)

                        pred_label[batch] = output / 255
                        
                        if conf < args.conf_threshold:
                            conf = 0.0
                            # pred_label[batch] = np.zeros(pred_label[batch].shape)
                        else:
                            conf = 1.0
                    else:  # empty ground truth label
                        image_cv = np.transpose(image[batch].copy(), (1, 2, 0))
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                        pred_label[batch] = pupilTrack(image_cv) / 255
                        conf = 0.0
                
                conf_list.append(conf)
                """plot segmentation map"""
                pred_label[batch] = pred_label[batch].astype(np.uint8)

                out_mask = np.stack((pred_label[batch], pred_label[batch], pred_label[batch]), axis=2)
                out_mask[:, :, 1] = 0
                cv2.imwrite(os.path.join(args.public_mask, folder_name + '_solution', image_name[batch][:-4] + '.png'), out_mask * 255)
                
            pbar.update()

        pbar.close()
        save_path = os.path.join(args.public_mask, folder_name + '_solution', 'conf.txt')
        np.savetxt(save_path,  conf_list, fmt='%.1f', delimiter=',')

def get_args():
    parser = argparse.ArgumentParser(
        description='2022 cv final project -- Pupil tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, default='../dataset', help='Number of epochs')
    parser.add_argument('--train_txt', type=str, default='../dataset/train.txt', help='path to dataset')
    parser.add_argument('--valid_txt', type=str, default='../dataset/valid.txt', help='path to dataset')

    parser.add_argument('--load', type=str, default='./model_seg.pth', help='Load model from a .pth file')
    parser.add_argument('--save_fig', default='./out_mask', help='path to save figures')
    parser.add_argument('--public_mask', default='./public_mask_deeplab', help='path to save pred masks')
    parser.add_argument('--subject', default='', help='Which subject you want to predict')
    parser.add_argument('--valid', action='store_true', help='Whether to use valid model!')

    parser.add_argument('--num_classes', type=int,default=2, help='Number of classes')
    parser.add_argument('--img_size', type=int,default=[384, 256], help='size of image', nargs='+')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model', type=str, default='deeplab', help='deeplab/sfnet/segformer/lawin/ensemble')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--conf_threshold', type=float, default=0.85, help='threshold for positive labels')
    parser.add_argument('--val_threshold', type=float, default=0.4, help='threshold for positive labels')
    parser.add_argument('--dice_weight', type=float, default=0.6, help='dice loss weight')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.valid:
        print('conf_th: %.2f | valid_th: %.2f' % (args.conf_threshold, args.val_threshold))
    else:
        print('conf_th: %.2f' % (args.conf_threshold))

    return args

if __name__ == '__main__':
    args = get_args()

    model_valid = models.resnet18(pretrained=True)
    model_valid.fc = nn.Linear(model_valid.fc.in_features, 1)
    model_valid.load_state_dict(torch.load('./model_valid.pth'))
    model_valid = model_valid.cuda()
    model_valid.eval()

    if args.model == 'deeplab':
        model_seg = DeepLab(backbone='resnet')
        model_seg.decoder.last_conv[8] = nn.Conv2d(256, args.num_classes, kernel_size=(1,1), stride=(1,1))
    elif args.model == 'segformer':
        model_seg = SegFormer(backbone='MiT-B2', num_classes=150)
        model_seg.decode_head.linear_pred = nn.Conv2d(768, args.num_classes, kernel_size=(1,1), stride=(1,1))
        model_seg = nn.DataParallel(model_seg)
    elif args.model == 'sfnet':
        model_seg = SFNet('ResNetD-18')
        model_seg.head.conv_seg = nn.Conv2d(128, args.num_classes, 1)
    elif args.model == 'lawin':
        model_seg = Lawin('MiT-B2')
        model_seg.decode_head.linear_pred = nn.Conv2d(512, args.num_classes, 1)
        model_seg = nn.DataParallel(model_seg)
    
    if args.load:
        print("Load pretrained model!!")
        model_seg.load_state_dict(torch.load(args.load))

    model_seg = model_seg.cuda()
    model_seg.eval()

    loss_func = DiceFocalLoss(args)
    # valid(args, model_seg, model_valid, loss_func)
    if args.subject:
        test(args, model_seg, model_valid)
    else:
        test_challenge(args, model_seg, model_valid)


