import numpy
import cv2

image_path = 'data/orig_images/image_20-07-2026_10-59-25.png'
image = cv2.imread(image_path, flags=cv2.COLOR_BGR2RGB)






cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()