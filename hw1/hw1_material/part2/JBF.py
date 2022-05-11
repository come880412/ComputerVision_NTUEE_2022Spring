from cmath import pi
from re import sub
import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s + 1
        # self.wndw_size = 3
        self.pad_w = 3*sigma_s

        scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
        scaleFactor_r = 1 / (2 * sigma_r **2 * 255 ** 2)

        # Generate guassian kernel
        self.spatial_kernel = [[np.exp(-(i**2+j**2) * scaleFactor_s) \
                                for i in range(-(self.wndw_size // 2), (self.wndw_size +1) // 2)]\
                                for j in range(-(self.wndw_size // 2), (self.wndw_size +1) // 2)]
        self.spatial_kernel = np.array(self.spatial_kernel) # (13, 13)

        # Generate look up table for range kernel
        self.LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)

    def joint_bilateral_filter(self, img, guidance):
        h, w, c = img.shape
        r = self.wndw_size // 2

        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        output = np.zeros(img.shape)
        ### TODO ###
        if guidance.ndim == 2:
            for i in range(r, r+h):
                for j in range(r, r+w):
                    range_kernel = self.LUT[np.abs(padded_guidance[i - r:i + r + 1, j - r:j + r + 1] - padded_guidance[i, j])] # (13, 13)
                    W = range_kernel * self.spatial_kernel
                    Pixel_value = padded_img[i-r:i+r+1, j-r:j+r+1]

                    output[i-r, j-r] = np.sum(W[:,:,np.newaxis] * Pixel_value, axis=(0,1)) / np.sum(W)
        elif guidance.ndim == 3:
            for i in range(r, r+h):
                for j in range(r, r+w):
                    range_kernel = self.LUT[np.abs(padded_guidance[i - r:i + r + 1, j - r:j + r + 1, 0] - padded_guidance[i, j, 0])] * \
                                   self.LUT[np.abs(padded_guidance[i - r:i + r + 1, j - r:j + r + 1, 1] - padded_guidance[i, j, 1])] * \
                                   self.LUT[np.abs(padded_guidance[i - r:i + r + 1, j - r:j + r + 1, 2] - padded_guidance[i, j, 2])] # (13, 13)
                    W = range_kernel * self.spatial_kernel
                    Pixel_value = padded_img[i-r:i+r+1, j-r:j+r+1]

                    output[i-r, j-r] = np.sum(W[:,:,np.newaxis] * Pixel_value, axis=(0,1)) / np.sum(W)

        return np.clip(output, 0, 255).astype(np.uint8)

    