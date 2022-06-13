import json

import os
import cv2
import numpy as np

import albumentations as A

if __name__ == '__main__':
    dataset = '../../dataset'
    save_dir = './visualization'

    os.makedirs(save_dir, exist_ok=True)

    image_path = '../dataset/public/S1/01/0.jpg'
    image = cv2.imread(image_path)

    mask = cv2.imread('../dataset/public/S1/01/0.png', 0)
    mask = mask[:, :, np.newaxis]

    cv2.imwrite(os.path.join(save_dir, 'image_ori.png'), image)
    cv2.imwrite(os.path.join(save_dir, 'mask_ori.png'), mask)

    transform = A.Resize(width=320, height=240)(image=image, mask=mask)
    image, mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_crop.png'), image)
    cv2.imwrite(os.path.join(save_dir, 'mask_crop.png'), mask)

    transform = A.HorizontalFlip(p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_hor.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_hor.png'), transformed_mask)

    transform = A.VerticalFlip(p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_ver.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_ver.png'), transformed_mask)

    transform = A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_bright.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_bright.png'), transformed_mask)

    transform = A.RandomRotate90(p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_rotate90.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_rotate90.png'), transformed_mask)

    transform = A.Transpose(p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_transpose.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_transpose.png'), transformed_mask)

    transform = A.ShiftScaleRotate(p=1)(image=image, mask=mask)
    transformed_image, transformed_mask = transform["image"], transform["mask"]
    cv2.imwrite(os.path.join(save_dir, 'image_shiftscalerotate.png'), transformed_image)
    cv2.imwrite(os.path.join(save_dir, 'mask_shiftscalerotate.png'), transformed_mask)
    
    
    # cv2.imshow('Extracted Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(image_name, img_width, img_height)
