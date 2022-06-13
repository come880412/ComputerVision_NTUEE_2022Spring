import os
import argparse
import numpy as np
import cv2

def get_args():
    parser = argparse.ArgumentParser(
        description='Visualize the output masks compared with ground truth',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', type=str, default='./train_mask', help='root directiry of predicted mask results, should contain S1 to S5')
    parser.add_argument('--output_path', type=str, default='./visualize_mask_diff', help='root directiry of output files')
    parser.add_argument('--gt_path', type=str, default='../dataset/public', help='path to dataset/public')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    result_path = args.input_path
    output_path = args.output_path
    gt_path = args.gt_path
    
    os.mkdir(output_path)
    for s in os.listdir(result_path):
        if 'S5' in s: continue# ignore S5
        s_o = s if len(s) < 3 else s[:2]
        os.mkdir(os.path.join(output_path, s_o))
        for num in os.listdir(os.path.join(result_path, s)):
            os.mkdir(os.path.join(output_path, s_o, num))
            for file in os.listdir(os.path.join(result_path, s, num)):
                if file == 'conf.txt': continue

                mask_result = cv2.imread(os.path.join(result_path, s, num, file), 0).astype(bool).astype(int)
                mask_gt = cv2.imread(os.path.join(gt_path, s_o, num, file), 0).astype(bool).astype(int)

                intersect = np.logical_and(mask_result, mask_gt).astype(np.uint8)
                sub_pred = mask_result - intersect
                sub_gt = mask_gt - intersect

                visualize = np.stack((sub_pred, intersect, sub_gt), axis=2)
                if np.sum(sub_pred) > 0 or np.sum(sub_gt) > 0:
                    cv2.imwrite(os.path.join(output_path, s_o, num, file), visualize.astype(np.uint8)*255)

    

