import numpy as np
import os
import random
import cv2
import tqdm

np.random.seed(2022) # 2022
random.seed(2022) # 2022

def train_val_split(data_path, save_path):
    subject = ['S1', 'S2', 'S3', 'S4']

    train_ratio = 0.9
    
    train_data = []
    valid_data = []
    for s in subject:
        subject_path = os.path.join(data_path, s)

        subject_clip = os.listdir(subject_path)
        subject_clip.sort()

        subject_len = len(subject_clip)

        total_index = np.random.choice(subject_len, subject_len, replace=False) + 1

        train_index = total_index[:int(subject_len*train_ratio)]
        valid_index = total_index[int(subject_len*train_ratio):]

        for idx in train_index:
            train_data.append('%s/%s' % (s, str(idx).zfill(2)))
        for idx in valid_index:
            valid_data.append('%s/%s' % (s, str(idx).zfill(2)))

    np.savetxt(os.path.join(save_path, 'train90.txt'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'valid90.txt'),  valid_data, fmt='%s', delimiter=',')

    print("Number of training subject: ", len(train_data))
    print("Number of valid subject: ", len(valid_data))

def valid(data_path, save_path):
    subject = ['S1', 'S2', 'S3', 'S4']

    valid = []
    non_valid = []
    for s in tqdm.tqdm(subject):
        for id in os.listdir(os.path.join(data_path, s)):
            image_len = len([name for name in os.listdir(os.path.join(data_path, s, id)) if name.endswith('.jpg')])

            for i in range(image_len):
                label_path = os.path.join(data_path, s, id, str(i) + '.png')

                label = cv2.imread(label_path, 0)
                if np.sum(label) == 0:
                    non_valid.append(['%s/%s/%s' % (s, id, str(i) + '.jpg'),'0'])
                else:
                    valid.append(['%s/%s/%s' % (s, id, str(i) + '.jpg'),'1'])

    np.savetxt(os.path.join(save_path, 'valid_eye.txt'),  valid, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(save_path, 'non_valid_eye.txt'),  non_valid, fmt='%s', delimiter=',')

    print("Number of valid eyes: ", len(valid))
    print("Number of Non_valid eyes: ", len(non_valid))


if __name__ == '__main__':
    data_path = '../../dataset/public'
    save_path = '../../dataset/'

    train_val_split(data_path, save_path)
    # valid(data_path, save_path)
    
