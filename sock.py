import socket
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time 
import datetime
import os
from resize import upscale
from obj_detect import find_objects

CAMERA_IP   = "192.168.2.101"   
CAMERA_PORT = 5001    
cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")

data_path = 'data'
resized_images_path = 'data/resized_images'
orig_images_path = 'data/orig_images'
obj_path = 'data/objects'
num = 256
need_resize = False
sleeep = 0
image_full = []

if not os.path.exists(obj_path):
    os.mkdir(obj_path)

if not os.path.exists(resized_images_path):
    os.mkdir(resized_images_path)

if not os.path.exists(orig_images_path):
    os.mkdir(orig_images_path)

start_time = time.time()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
for j in range(num):
    time.sleep(sleeep)
    strings = []
    sock.settimeout(0.1)
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for i in range(12):
        data, addr = sock.recvfrom(65535)      # сеть
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)   # вычисления
        strings.extend(bright)

    image_full.extend(strings)
    end_time_cycle = time.time()
    cycle_time = (end_time_cycle - start_time) / (j + 1)
    if j % 100 == 0:
        print(f'Epochs: {j + 1}, Time: {cycle_time}')

sock.close()
image_full = np.array(image_full, dtype=np.uint8).reshape(num, -1, 3)
image_full = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
print(image_full.shape)

date = datetime.datetime.now()
formatted_string_ru = date.strftime("%d-%m-%Y_%H-%M-%S")

str_image = f"image_{formatted_string_ru}.png"

str_orig = os.path.join(orig_images_path,str_image )
cv2.imwrite(str_orig, image_full)
find_objects(image_full, str_image)

if need_resize:
    image_resized = upscale(image_full)
    str_resized = os.path.join(resized_images_path, f"image_resized_{formatted_string_ru}.png")
    cv2.imwrite(str_resized, image_resized)

end_time = time.time()
print('Конец работы:', (end_time - start_time))
