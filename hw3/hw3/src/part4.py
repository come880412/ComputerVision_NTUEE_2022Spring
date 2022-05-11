import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def matches(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc_a, desc_b, k=2)

    good_matches = []
    for match_1, match_2 in matches:
        if match_1.distance < 0.8 * match_2.distance:
            good_matches.append(match_1)
    
    #filter good matching keypoints 
    good_kp_a = []
    good_kp_b = []

    for match in good_matches:
        good_kp_a.append(kp_a[match.queryIdx].pt) # keypoint in image A
        good_kp_b.append(kp_b[match.trainIdx].pt) # matching keypoint in image B
    return np.array(good_kp_a).astype(np.int), np.array(good_kp_b).astype(np.int)

def transform_with_homography(H, points):
    ones = np.ones((points.shape[0], 1))
    points = np.concatenate((points, ones), axis=1)
    transformed_points = H.dot(points.T)
    transformed_points = transformed_points / (transformed_points[2,:][np.newaxis, :])
    transformed_points = transformed_points[0:2,:].T

    return transformed_points

def compute_outlier(H, points_a, points_b, threshold=3):
    outliers_count = 0

    points_img_b_transformed = transform_with_homography(H, points_b)

    x = points_a[:, 0]
    y = points_a[:, 1]
    x_hat = points_img_b_transformed[:, 0]
    y_hat = points_img_b_transformed[:, 1]
    distance = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in distance:
        if dis > threshold:
            outliers_count += 1
    return outliers_count

def ransac_for_homography(matches_1, matches_2):

    all_matches = matches_1.shape[0]
    # RANSAC parameters
    prob_success = 0.99
    sample_points_size = 5
    ratio_of_outlier = 0.5
    N = int(np.log(1.0 - prob_success)/np.log(1 - (1 - ratio_of_outlier)**sample_points_size))

    lowest_outlier = all_matches # Worst case: all the points are outliers
    best_H = None

    for i in range(N):
        rand_index = np.random.choice(all_matches, sample_points_size, replace=False)
        H = solve_homography(matches_2[rand_index], matches_1[rand_index])
        outliers_count = compute_outlier(H, matches_1, matches_2)
        if outliers_count < lowest_outlier:
            best_H = H
            lowest_outlier = outliers_count

    return best_H

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        matches_1, matches_2 = matches(im1, im2)

        # TODO: 2. apply RANSAC to choose best H
        H = ransac_for_homography(matches_1, matches_2)

        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H)

        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 'b')
    out = dst

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)