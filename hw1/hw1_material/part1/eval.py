##################################
### DO NOT modify this file!!! ###
##################################

import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def main():
    parser = argparse.ArgumentParser(description = 'evaluation function of Difference of Gaussian')
    parser.add_argument('--threshold', default = 5.0, type=float, help = 'threshold value for feature selection')
    parser.add_argument('--image_path', default = './testdata/1.png', help = 'path to input image')
    parser.add_argument('--gt_path', default = './testdata/1_gt.npy', help = 'path to ground truth .npy')
    args = parser.parse_args()

    img = cv2.imread(args.image_path, 0).astype(np.float64)

    # create DoG class
    DoG = Difference_of_Gaussian(args.threshold)
    
    # find keypoint from DoG and sort it
    keypoints = DoG.get_keypoints(img)

    # read GT
    keypoints_gt = np.load(args.gt_path)

    if keypoints.shape[0] != keypoints_gt.shape[0]:
        print('[Error] Number of KeyPoints mismatches. Yours: %d, GT: %d'%
              (keypoints.shape[0], keypoints_gt.shape[0]))
    else:
        for (x, y), (x_gt, y_gt) in zip(keypoints, keypoints_gt):
            if x != x_gt or y != y_gt:
                print('[Error] KeyPoint mismatches. Yours: (%d, %d), GT: (%d, %d)'%
                        (x, y, x_gt, y_gt))
                return None
        print('[Info] All keypoints match.')

if __name__ == '__main__':
    main()