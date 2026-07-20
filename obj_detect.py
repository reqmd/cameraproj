import numpy as np
import cv2
import os

image_path = 'data/orig_images/image_20-07-2026_15-19-22.png'
obj_path = 'data/objects'

image = cv2.imread(image_path, flags=cv2.COLOR_BGR2GRAY)
if not os.path.exists(obj_path):
    os.mkdir(obj_path)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()