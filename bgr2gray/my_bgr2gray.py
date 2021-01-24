import cv2
import numpy as np

def my_bgr2gray(BGR) :
    B = BGR[:, :, 0]
    G = BGR[:, :, 1]
    R = BGR[:, :, 2]
    grayScale = B * 0.1140 + G * 0.5870 + R * 0.2989
    grayScale = grayScale.astype(np.uint8)
    return grayScale;

bgr = cv2.imread('./img.jpg',cv2.IMREAD_COLOR)
cv2.imshow('grab2bgr',my_bgr2gray(bgr))
cv2.waitKey(0)