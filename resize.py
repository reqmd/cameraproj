import numpy as np
import cv2

def upscale(image, K = 10):
    image = cv2.resize(image, None, fx=1, fy = K, interpolation=cv2.INTER_CUBIC)
    return image

