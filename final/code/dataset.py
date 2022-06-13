import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A
import cv2
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image

label_dict = {0:"background", 1:"pupil"}

class TEyeD_pupil_seg_data(Dataset):
    def __init__(self, args, data, isTrain=True):
        self.data = data
        self.root = args.root
        self.isTrain = isTrain
        self.image_size = args.img_size

        self.data_info = []

        if isTrain:
            self.transform = A.Compose([
                                A.Resize(width=self.image_size[0], height=self.image_size[1]),
                                A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                                A.HorizontalFlip(p=0.3),
                                A.VerticalFlip(p=0.3),
                                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                                
                                A.Equalize(p=0.3),
                                # A.HueSaturationValue(p=0.4),
                                A.Sharpen(p=0.3),
                                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
                            ])
        else:
            self.transform = None

        for image_name in self.data:
            label_path = os.path.join(self.root, 'TEyeD', 'masks', image_name)
            image_path = os.path.join(self.root, 'TEyeD', 'images', image_name)

            self.data_info.append([image_path, label_path])
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_info[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, 3)

        mask = cv2.imread(label_path, 0)
        mask = mask[:, :, np.newaxis] / 255.0

        # self.visualization(image, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = image.transpose((2, 0, 1)) # (3, H, W)
        mask = mask.transpose((2, 0, 1)) # (1, H, W)
        
        image = torch.from_numpy((image.copy()))
        image = image.float() / 255.0
        mask = torch.from_numpy((mask.copy()))

        return image, mask
    def visualization(self, image, mask):
        transform = A.Resize(height=384, width=384)(image=image, mask=mask)
        image, mask = transform["image"], transform["mask"]

        transform = A.HorizontalFlip(p=1)(image=image, mask=mask)
        hor_image, hor_mask = transform["image"], transform["mask"]

        transform = A.VerticalFlip(p=1)(image=image, mask=mask)
        ver_image, ver_mask = transform["image"], transform["mask"]

        transform = A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)(image=image, mask=mask)
        bright_image, bright_mask = transform["image"], transform["mask"]

        transform = A.RandomRotate90(p=1)(image=image, mask=mask)
        rotate_image, rotate_mask = transform["image"], transform["mask"]

        transform = A.Transpose(p=1)(image=image, mask=mask)
        transpose_image, transpose_mask = transform["image"], transform["mask"]

        transform = A.ShiftScaleRotate(p=1)(image=image, mask=mask)
        shift_image, shift_mask = transform["image"], transform["mask"]

        plt.figure(figsize=(18,12))
        plt.subplot(4,4,1)
        plt.title("image_ori")
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(4,4,2)
        plt.title("mask_ori")
        plt.imshow(mask)
        plt.axis('off')
        plt.subplot(4,4,3)
        plt.title("imageped")
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(4,4,4)
        plt.title("maskped")
        plt.imshow(mask)
        plt.axis('off')
        plt.subplot(4,4,5)
        plt.title("image_vertial_flip")
        plt.imshow(ver_image)
        plt.axis('off')
        plt.subplot(4,4,6)
        plt.title("mask_vertial_flip")
        plt.imshow(ver_mask)
        plt.axis('off')
        plt.subplot(4,4,7)
        plt.title("image_horizontal_flip")
        plt.imshow(hor_image)
        plt.axis('off')
        plt.subplot(4,4,8)
        plt.title("mask_horizontal_flip")
        plt.imshow(hor_mask)
        plt.axis('off')
        plt.subplot(4,4,9)
        plt.title("image_rotate")
        plt.imshow(rotate_image)
        plt.axis('off')
        plt.subplot(4,4,10)
        plt.title("mask_rotate")
        plt.imshow(rotate_mask)
        plt.axis('off')
        plt.subplot(4,4,11)
        plt.title("image_rotateshift")
        plt.imshow(shift_image)
        plt.axis('off')
        plt.subplot(4,4,12)
        plt.title("mask_rotateshift")
        plt.imshow(shift_mask)
        plt.axis('off')
        plt.subplot(4,4,13)
        plt.title("image_bright")
        plt.imshow(bright_image)
        plt.axis('off')
        plt.subplot(4,4,14)
        plt.title("mask_bright")
        plt.imshow(bright_mask)
        plt.axis('off')
        plt.subplot(4,4,15)
        plt.title("image_transpose")
        plt.imshow(transpose_image)
        plt.axis('off')
        plt.subplot(4,4,16)
        plt.title("mask_transpose")
        plt.imshow(transpose_mask)
        plt.axis('off')
        plt.show()

class ganzin_pupil_seg_data(Dataset):
    def __init__(self, args, isTrain=True, nonvalid=False):
        self.root = args.root
        self.isTrain = isTrain
        self.image_size = args.img_size

        self.data_info = []
        data_valid = np.loadtxt(os.path.join(args.root, 'valid_eye.txt'), delimiter=',', dtype=np.str)
        data_nonvalid = np.loadtxt(os.path.join(args.root, 'non_valid_eye.txt'), delimiter=',', dtype=np.str)

        if isTrain:
            data = np.loadtxt(args.train_txt, delimiter=',', dtype=np.str)
            self.transform = A.Compose([
                                A.Resize(width=self.image_size[0], height=self.image_size[1]),
                                A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                                A.HorizontalFlip(p=0.3),
                                A.VerticalFlip(p=0.3),
                                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                                
                                A.Equalize(p=0.3),
                                # A.HueSaturationValue(p=0.4),
                                A.Sharpen(p=0.3),
                                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.3),
                                
                            ])
        else:
            data = np.loadtxt(args.valid_txt, delimiter=',', dtype=np.str)
            # self.transform = A.Equalize(p=1)
            self.transform = None

        for info in data_valid:
            image_info, _ = info
            image_subject = image_info.split('/')[0] + '/' + image_info.split('/')[1]
            image_name = image_info.split('/')[2][:-4]
            if image_subject in data:
                label_path = os.path.join(self.root, 'public', image_subject, image_name + '.png')
                image_path = os.path.join(self.root, 'public', image_subject, image_name + '.jpg')
                self.data_info.append([image_path, label_path])
        if nonvalid:
            for info in data_nonvalid:
                image_info, _ = info
                image_subject = image_info.split('/')[0] + '/' + image_info.split('/')[1]
                image_name = image_info.split('/')[2][:-4]
                if image_subject in data:
                    label_path = os.path.join(self.root, 'public', image_subject, image_name + '.png')
                    image_path = os.path.join(self.root, 'public', image_subject, image_name + '.jpg')
                    self.data_info.append([image_path, label_path])

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, label_path = self.data_info[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, 3)

        mask = cv2.imread(label_path, 0)[:, :, np.newaxis].astype(np.float)
        if np.sum(mask) != 0:
            mask = mask / np.max(mask)

        # self.visualization(image, mask * 255)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = image.transpose((2, 0, 1)) # (3, H, W)
        mask = mask.transpose((2, 0, 1)) # (1, H, W)
        
        image = torch.from_numpy((image.copy()))
        image = image.float() / 255.0
        mean = torch.as_tensor([0.38314])
        std = torch.as_tensor([0.31113])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        mask = torch.from_numpy((mask.copy()))

        return image, mask
    def visualization(self, image, mask):
        image_ori = image.copy()
        mask_ori = mask.copy()

        transform = A.Equalize(p=1)(image=image, mask=mask)
        image, mask = transform["image"], transform["mask"]

        transform = A.HorizontalFlip(p=1)(image=image, mask=mask)
        hor_image, hor_mask = transform["image"], transform["mask"]

        transform = A.VerticalFlip(p=1)(image=image, mask=mask)
        ver_image, ver_mask = transform["image"], transform["mask"]

        transform = A.Rotate(limit=30, p=1.0, border_mode=cv2.BORDER_CONSTANT)(image=image, mask=mask)
        rotate_image, rotate_mask = transform["image"], transform["mask"]

        transform = A.HueSaturationValue(hue_shift_limit=0.0, p=1.0)(image=image, mask=mask)
        hue_sat_image, hue_sat_mask = transform["image"], transform["mask"]

        transform = A.Sharpen(p=1.0)(image=image, mask=mask)
        sharpen_image, sharpen_mask = transform["image"], transform["mask"]

        transform = A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)(image=image, mask=mask)
        bright_image, bright_mask = transform["image"], transform["mask"]

        transform = A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1.0, border_mode=cv2.BORDER_CONSTANT)(image=image, mask=mask)
        shift_image, shift_mask = transform["image"], transform["mask"]

        plt.figure(figsize=(18,12))
        plt.subplot(5,4,1)
        plt.title("image_ori")
        plt.imshow(image_ori)
        plt.axis('off')

        plt.subplot(5,4,2)
        plt.title("mask_ori")
        plt.imshow(mask_ori)
        plt.axis('off')

        plt.subplot(5,4,3)
        plt.title("image equalization")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(5,4,4)
        plt.title("mask")
        plt.imshow(mask)
        plt.axis('off')

        plt.subplot(5,4,5)
        plt.title("image_vertial_flip")
        plt.imshow(ver_image)
        plt.axis('off')

        plt.subplot(5,4,6)
        plt.title("mask_vertial_flip")
        plt.imshow(ver_mask)
        plt.axis('off')

        plt.subplot(5,4,7)
        plt.title("image_horizontal_flip")
        plt.imshow(hor_image)
        plt.axis('off')

        plt.subplot(5,4,8)
        plt.title("mask_horizontal_flip")
        plt.imshow(hor_mask)
        plt.axis('off')

        plt.subplot(5,4,9)
        plt.title("image_rotate")
        plt.imshow(rotate_image)
        plt.axis('off')
        plt.subplot(5,4,10)
        plt.title("mask_rotate")
        plt.imshow(rotate_mask)
        plt.axis('off')

        plt.subplot(5,4,11)
        plt.title("image_shiftscale")
        plt.imshow(shift_image)
        plt.axis('off')

        plt.subplot(5,4,12)
        plt.title("mask_shiftscale")
        plt.imshow(shift_mask)
        plt.axis('off')

        plt.subplot(5,4,13)
        plt.title("image_brightness")
        plt.imshow(bright_image)
        plt.axis('off')

        plt.subplot(5,4,14)
        plt.title("mask_brightness")
        plt.imshow(bright_mask)
        plt.axis('off')

        plt.subplot(5,4,15)
        plt.title("image_hue_sat")
        plt.imshow(hue_sat_image)
        plt.axis('off')

        plt.subplot(5,4,16)
        plt.title("mask_hue_sat")
        plt.imshow(hue_sat_mask)
        plt.axis('off')

        plt.subplot(5,4,17)
        plt.title("image_sharpen")
        plt.imshow(sharpen_image)
        plt.axis('off')

        plt.subplot(5,4,18)
        plt.title("mask_sharpen")
        plt.imshow(sharpen_mask)
        plt.axis('off')
        plt.show()

class ganzin_pupil_valid_data(Dataset):
    def __init__(self, args, data, isTrain=True):
        self.root = args.root
        self.isTrain = isTrain
        self.img_size = args.img_size

        self.data_info = []

        if isTrain:
            self.transform = transforms.Compose([
                                transforms.RandomEqualize(p=1.0), 
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.RandomVerticalFlip(p=0.3),
                                transforms.RandomApply(torch.nn.ModuleList([
                                    transforms.ColorJitter(brightness=0.3, saturation=0.3, contrast=0.3),
                                ]), p=0.3),
                                transforms.RandomAffine(degrees=20, translate=(0.2,0.2), scale=(0.5, 1.5)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.38314), (0.31113))
                            ])
        else:
            self.transform = transforms.Compose([
                                # transforms.RandomEqualize(p=1.0), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.38314), (0.31113))
                            ])

        
        for info in data:
            image_info, label = info
            image_path = os.path.join(self.root, 'public', image_info)

            self.data_info.append([image_path, float(label)])

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        image_path, label = self.data_info[index] # 0: non_valid, 1:valid
        label = np.array(label).astype(np.float)
        image = Image.open(image_path).convert('RGB')

        return self.transform(image), torch.FloatTensor(label)

class ganzin_pupil_seg_public_data(Dataset):
    def __init__(self, data_path, subject='S5'):
        self.root = os.path.join(data_path, 'public')
        self.data_path = []
        self.data_info = []
        self.equalize = A.Equalize(p=1)
        
        if subject:
            ID_list = [str(id).zfill(2) for id in range(1, 27)]
            for id in ID_list:
                data_len = len([name for name in os.listdir(os.path.join(self.root, subject, id)) if name.endswith('.jpg')])
                for num in range(data_len):
                    self.data_path.append(os.path.join(self.root, subject, id, f'{num}.jpg'))
                    self.data_info.append([subject, id, f'{num}.jpg'])
        else:
            image_names = sorted(os.listdir(data_path))

            for image_name in image_names:
                self.data_path.append(os.path.join(data_path, image_name))
                self.data_info.append(image_name)

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        image_path = self.data_path[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, 3)

        # self.visualization(image)
        
        transformed = self.equalize(image=image)
        image= transformed["image"]

        image = image.transpose((2, 0, 1)) # (3, H, W)
        
        image = torch.from_numpy((image.copy()))
        image = image.float() / 255.0
        mean = torch.as_tensor([0.38314])
        std = torch.as_tensor([0.31113])
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image, self.data_info[index]

    def visualization(self, image):
        transform = A.Equalize(p=1)(image=image)
        image_equal = transform["image"]

        plt.figure(figsize=(18,12))
        plt.subplot(1,2,1)
        plt.title("image_ori")
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title("image equalization")
        plt.imshow(image_equal)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    pass
    # Norm()
    # parser = argparse.ArgumentParser(
    #     description='Train the UNet on images and target masks',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--root', type=str, default='../../dataset', help='Number of epochs')
    # parser.add_argument('--epochs', type=int,default=5, help='Number of epochs')
    # parser.add_argument('--img_size', type=int,default=896, help='size of image')
    # parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    # parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', type=float, default=0.1,help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--split_seed', type=int, default=2022, help='')
    # args = parser.parse_args()
    # data_list = os.listdir(os.path.join(args.root, 'Train_Images'))
    # train_data_list, valid_data_list = train_test_split(data_list, random_state=args.split_seed, test_size=args.validation)

    # mean = torch.as_tensor([0.827, 0.621, 0.769])
    # std = torch.as_tensor([0.168, 0.302, 0.190])
    # if mean.ndim == 1:
    #     mean = mean.view(-1, 1, 1)
    # if std.ndim == 1:
    #     std = std.view(-1, 1, 1)

    # valid_data = cancer_seg_data(args, train_data_list, isTrain=True)
    # valid_loader = DataLoader(valid_data, batch_size=1, num_workers=0, shuffle=True, pin_memory=True)
    # for image, mask in valid_loader:
    #     mask = mask[:,0,:,:]
    #     image = image.cpu().detach()
    #     image = ((image * std) + mean) * 255
    #     image = image.numpy()
        # image = image.astype(np.uint8)
        # for batch_size in range(len(image)):
        #     plt.figure(figsize=(18,12))
        #     plt.subplot(1,2,1)
        #     plt.title("image")
        #     plt.imshow(image[batch_size].transpose((1, 2, 0)))
        #     plt.subplot(1,2,2)
        #     plt.title("label")
        #     plt.imshow(mask[batch_size])
        #     plt.show()

    # data_loader = DataLoader(data, batch_size=2, shuffle=True, drop_last=False)
    # iter_data = iter(data_loader)
    # image, mask = iter_data.next()
    # print(image.shape, mask.shape)