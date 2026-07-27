import numpy as np
import socket
import cv2
import datetime
import os
 
CAMERA_IP   = "192.168.2.101"
CAMERA_PORT = 5001
WIDTH = 2716
X0, X1 = 175, WIDTH - 1000
 
cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.1)
background = np.load('background.npy').astype(np.int16)
 
date = datetime.datetime.now()
session_name = date.strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(f"data/sessions/{session_name}", exist_ok=True)
 
TH = np.array([25, 25, 25])        # пороги по каналам
MIN_FG_PIXELS = 400                 # сколько отклонившихся пикселей в строке = "объект есть"
GAP_ROWS = 3                       # столько подряд фоновых строк = объект закончился
MAX_ROWS = 64                    # предохранитель от бесконечного объекта
ALPHA = 0.01
 
def read_line():
    strings = []
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for _ in range(12):
        data, _ = sock.recvfrom(65535)
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)
        strings.extend(bright)
    rgb = np.array(strings, dtype=np.uint8).reshape(-1, 3)
    return rgb[X0:X1]
 
 
def row_has_object(row):
    """True, если строка заметно отличается от фона."""
    diff = np.abs(row.astype(np.int16) - background)
    fg = (diff > TH).any(axis=1)  
    print(fg.sum())                      
    return fg.sum() >= MIN_FG_PIXELS
 
 
def find_obj(image, session_name, threshold=(25, 25, 25), obj_counter=[0]):
    """image — собранный объект (строки, ширина, 3). Выделяет и сохраняет объекты."""
    r_th, g_th, b_th = threshold
    diff = np.abs(image.astype(np.int16) - background)
 
    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255
 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
 
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
 
    MIN_AREA = 2000
    found = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        crop = image[y:y+h, x:x+w]
        fname = f"data/sessions/{session_name}/object_{obj_counter[0]}.png"
        cv2.imwrite(fname, crop[:, :, ::-1])         
        obj_counter[0] += 1
        found += 1
    if found:
        print("сохранено объектов:", found)
 
 
# ---------------- основной цикл: конечный автомат ----------------
collecting = False
obj_rows = []
gap = 0
 
while True:
    try:
        row = read_line()
    except socket.timeout:
        continue
 
    if row_has_object(row):
        obj_rows.append(row)
        collecting = True
        gap = 0
        if len(obj_rows) > MAX_ROWS:          # слишком длинный — сбрасываем на всякий
            obj_rows = []
            collecting = False
    else:
        background = ((1 - ALPHA) * background + ALPHA * row).astype(np.int16)
        if collecting:
            gap += 1
            obj_rows.append(row)              # добавляем и фоновые строки в "хвост"
            if gap >= GAP_ROWS:               # объект точно закончился
                image = np.stack(obj_rows)    # (строки, ширина, 3)
                find_obj(image, session_name)
                obj_rows = []
                collecting = False
                gap = 0