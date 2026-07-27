import numpy as np
import cv2
import os
import shutil

def find_objects(image, image_name, threshhold = [25, 25, 25], need_imshow = True):
    r_th, g_th, b_th = threshhold
    H, W, _ = image.shape
    image = image[0:H, 200:W - 600]
    mask = np.zeros((H, W), dtype=np.uint8)
    background = np.median(image[:20], axis=0).astype(np.int16)
    # np.save('background.npy', background)
    
    diff = np.abs(image.astype(np.int16) - background)    

    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    MIN_AREA = 2000
    objects = []
    for i in range(1, n):                      
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:                    
            continue
        crop = image[y:y+h, x:x+w]             
        obj_mask = (labels[y:y+h, x:x+w] == i) 
        cx, cy = centroids[i]
        objects.append({'crop': crop, 'mask': obj_mask,
                        'bbox': (x, y, w, h), 'area': area, 'center': (cx, cy)})

    if os.path.exists(f'data/objects/{image_name}'):
        shutil.rmtree(f'data/objects/{image_name}')
    os.mkdir(f'data/objects/{image_name}')
    for idx, obj in enumerate(objects):
        cv2.imwrite(f"data/objects/{image_name}/object_{idx}.png", obj['crop'])

    print("найдено объектов:", len(objects))
    if need_imshow:
        vis = image.copy()
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)










# image_name = 'image_20-07-2026_15-20-13'
# image_path = f'data/orig_images/{image_name}.png'
# obj_path = 'data/objects'

# image = cv2.imread(image_path, flags=cv2.COLOR_BGR2GRAY)
# if not os.path.exists(obj_path):
#     os.mkdir(obj_path)

# H, W, C = image.shape
# image = image[0:H, 200:W - 600]

# r_th = 25
# g_th = 25
# b_th = 25

# background = np.median(image[:20], axis=0).astype(np.int16)
# diff = np.abs(image.astype(np.int16) - background)    

# mask = ((diff[:, :, 0] > r_th) |
#         (diff[:, :, 1] > g_th) |
#         (diff[:, :, 2] > b_th)).astype(np.uint8) * 255

# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
#                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

# n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

# MIN_AREA = 1200

# objects = []
# for i in range(1, n):                      
#     x, y, w, h, area = stats[i]
#     if area < MIN_AREA:                    
#         continue

#     crop = image[y:y+h, x:x+w]             
#     obj_mask = (labels[y:y+h, x:x+w] == i) 
#     cx, cy = centroids[i]

#     objects.append({'crop': crop, 'mask': obj_mask,
#                     'bbox': (x, y, w, h), 'area': area, 'center': (cx, cy)})

# if os.path.exists(f'data/objects/{image_name}'):
#     shutil.rmtree(f'data/objects/{image_name}')

# os.mkdir(f'data/objects/{image_name}')

# for idx, obj in enumerate(objects):
#     cv2.imwrite(f"data/objects/{image_name}/object_{idx}.png", obj['crop'])

# print("найдено объектов:", len(objects))

# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# vis = image.copy()
# for obj in objects:
#     x, y, w, h = obj['bbox']
#     cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)