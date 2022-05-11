import numpy as np
import cv2
import os

def PixelIsExtremum(first_image, second_image, third_image, threshold):
    center_pixel = second_image[1, 1]
    if abs(center_pixel) > threshold:
        if center_pixel > 0:
            first_cmp = (center_pixel >= first_image).all()
            third_cmp = (center_pixel >= third_image).all()
            second_cmp = ((center_pixel >= second_image[0, :]).all() and \
                           center_pixel >= second_image[1, 0] and \
                            center_pixel >= second_image[1,2] and \
                            (center_pixel >= second_image[2, :]).all())
            if first_cmp and third_cmp and second_cmp:
                return True

        elif center_pixel < 0:
            first_cmp = (center_pixel <= first_image).all()
            third_cmp = (center_pixel <= third_image).all()
            second_cmp = ((center_pixel <= second_image[0, :]).all() and \
                           center_pixel <= second_image[1, 0] and \
                           center_pixel <= second_image[1,2] and \
                           (center_pixel <= second_image[2, :]).all())
            if first_cmp and third_cmp and second_cmp:
                return True
    return False

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        h, w = image.shape
        gaussian_images = []
        
        for octave_idx in range(self.num_octaves):
            gaussian_images_per_octave = []
            gaussian_images_per_octave.append(image)
            
            for images_idx in range(self.num_DoG_images_per_octave):
                gaussian_blur_image = cv2.GaussianBlur(image, (0,0), sigmaX=self.sigma ** (images_idx + 1), sigmaY=self.sigma ** (images_idx + 1))
                gaussian_images_per_octave.append(gaussian_blur_image)
            gaussian_images.append(gaussian_images_per_octave)
            image = gaussian_images_per_octave[-1]
            image = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for octave_idx in range(self.num_octaves):
            dog_images_per_octave = []
            gaussian_images_per_octave = gaussian_images[octave_idx]
            for images_idx in range(self.num_guassian_images_per_octave - 1):
                dog_images_per_octave.append(cv2.subtract(gaussian_images_per_octave[images_idx + 1], gaussian_images_per_octave[images_idx]))
            dog_images.append(dog_images_per_octave)

        
        # 顯示圖片
        # cv2.imshow('ori_image', gaussian_images[0][0].astype(np.uint8))
        cv2.imshow('ori_downsampled_image', gaussian_images[1][0].astype(np.uint8))
        for octave_idx in range(len(dog_images)):
            dog_images_per_octave = dog_images[octave_idx]
            for idx, dog_image in enumerate(dog_images_per_octave):
                min, max = np.min(dog_image), np.max(dog_image)
                dog_image_save = (dog_image - min) / (max - min)
                cv2.imwrite('./report_img/DoG_%d-%d.png' % (octave_idx+1, idx+1), dog_image_save * 255)
                # cv2.imshow('DoG_%d-%d' % (octave_idx+1, idx+1), dog_image.astype(np.uint8))

        # 按下任意鍵則關閉所有視窗
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for octave_idx, dog_images_per_octave in enumerate(dog_images):
            for images_idx in range(len(dog_images_per_octave) -2):
                first_image, second_image, third_image = dog_images_per_octave[images_idx], dog_images_per_octave[images_idx+1], dog_images_per_octave[images_idx+2]

                for i in range(1, first_image.shape[0] - 2):
                    for j in range(1, first_image.shape[1] - 2):
                        if PixelIsExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], self.threshold):
                            
                            if octave_idx == 0:
                                x, y = i, j
                            else:
                                x, y = 2*i, 2*j
                            keypoints.append([x, y])
        keypoints = np.array(keypoints)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=1)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
