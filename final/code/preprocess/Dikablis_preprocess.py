import cv2
import numpy as np
from tqdm import tqdm

import os
import argparse
import random
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

'''
Only process on Dikablis dataset

output directory tree will looks like this:

args.root/
        ├─TEyeDS (Original dataset)
        |   └─Dikablis
        |       ├─ANNOTATIONS
        |       ├─VIDEOS
        |       ├─Readme.txt
        |
        └─TEyeD/ (Will create this directory automatically)
        └─TEyeD_valid/ (Will create this directory automatically)
'''

np.random.seed(2022)
random.seed(2022)
def data_process(args, video_path, label_path):
    video_list = os.listdir(video_path)
    video_list.sort()

    tqdm.write(f'Number of videos: {len(video_list)}')
    save_path = os.path.join('../../dataset', "TEyeD")
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "masks"), exist_ok=True)

    # BROKEN FILE
    broken = []
    broken.append("DikablisSS_10_1.mp4")
    with open("./others/pupil_seg_broken.txt", 'r') as p:
        with open("./others/iris_seg_broken.txt", 'r') as i:
            with open("./others/lid_seg_broken.txt", 'r') as l:
                for line in p.readlines():
                    broken.append(line.strip())
                for line in i.readlines():
                    broken.append(line.strip())
                for line in l.readlines():
                    broken.append(line.strip())

    
    source_image_idx = 0
    pupil_seg_idx = 0
    for video_name in tqdm(video_list):
        if video_name in broken:
            continue    
        '''
        image shape: (288, 384, 3) in Dikablis
        '''

        # Source video
        video = cv2.VideoCapture(os.path.join(video_path, video_name))
        success = True

        source_image_count = 0
        while(success):
            # OUTPUT DIR

            success, frame = video.read()
            if not success:
                break

            cv2.imwrite(os.path.join(save_path, "images", f'{str(source_image_idx).zfill(7)}.png'), frame)

            source_image_count += 1
            source_image_idx += 1

            if source_image_count == args.fpv: # reach desired frames per video
                break
        
        # pupil 2D seg
        video = cv2.VideoCapture(os.path.join(label_path,f'{video_name}pupil_seg_2D.mp4'))
        success = True
        while(success):
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame[frame <= 128] = 0
            frame[frame > 128] = 255
            im = Image.fromarray(frame)
            im.save(os.path.join(save_path, "masks",f'{str(pupil_seg_idx).zfill(7)}.png'))
            pupil_seg_idx += 1

            if pupil_seg_idx == source_image_idx:
                break

    print('------Statistics--------')
    print('num_data: ', pupil_seg_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../../neurobit/dataset', help='Root directory')
    parser.add_argument('--fpv', type=int, default=1500, help='How many frames per video you want to store.')
    args = parser.parse_args()

    # PATH
    video_path = os.path.join(args.root, "TEyeDS/Dikablis/VIDEOS")
    label_path = os.path.join(args.root, "TEyeDS/Dikablis/ANNOTATIONS")

    data_process(args, video_path, label_path)