import numpy as np
import cv2
from cv2 import aruco
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # TODO: find homography per frame and apply backward warp
    frame_idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {:d}'.format(frame_idx))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            # TODO: 1.find corners with aruco
            # function call to aruco.detectMarkers()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
            points = np.array(corners[0]).astype(np.int)[0]

            # TODO: 2.find homograpy
            # function call to solve_homography()
            H = solve_homography(ref_corns, points)

            # TODO: 3.apply backward warp
            # function call to warping()
            frame = warping(ref_image, frame, H, direction='b')

            videowriter.write(frame)
            frame_idx += 1

        else:
            break

    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/dog.png'
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)