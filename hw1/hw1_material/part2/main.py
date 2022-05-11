import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

from ctypes import *


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    guidance_images = [img_gray]
    W = []
    ### TODO ###
    with open(args.setting_path, 'r') as f:
        for idx, line in enumerate(f):
            info = line.strip().split(',')
            if idx > 0 and idx < 6:
                info[:] = [float(x) for x in info]
                W.append(info)
            elif idx == 6:
                sigma_s = int(info[1])
                sigma_r = float(info[3])
    
    for i in range(len(W)):
        guidance = img_rgb[:, :, 0] * W[i][0] + img_rgb[:, :, 1] * W[i][1] + img_rgb[:, :, 2] * W[i][2]
        guidance_images.append(guidance)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    # guidance_image = guidance_images[-1]
    # gf_out = JBF.joint_bilateral_filter(img_rgb, guidance_image)
    # gf_out = cv2.cvtColor(gf_out, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('./report_img/Lowest_cost_Grayscale_1.png', guidance_image)
    # cv2.imwrite('./report_img/Lowest_cost_FilteredImage_1.png', gf_out)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    for i in range(len(guidance_images)):
        guidance_image = guidance_images[i]
        gf_out = JBF.joint_bilateral_filter(img_rgb, guidance_image)
        cost = np.sum(np.abs(bf_out.astype('int32')-gf_out.astype('int32')))

        if i == 0:
            print('cv2.COLOR_BGR2GRAY cost: ', cost)
        else:
            print('R*%.1f+G*%.1f+B*%.1f: ' % (W[i-1][0], W[i-1][1], W[i-1][2]), cost)


if __name__ == '__main__':
    main()