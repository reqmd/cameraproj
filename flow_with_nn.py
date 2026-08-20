import threading
import queue
import torch.nn as nn
import numpy as np
import socket
import cv2
import datetime
import os
from test import eval
from rknn import inference_rknn
import torch
from torchvision import transforms
import time

CAMERA_IP   = "192.168.2.101"
CAMERA_PORT = 5001
WIDTH = 2716
X0, X1 = 175, WIDTH - 1000

obj_queue = queue.Queue(maxsize=20)   
cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.1)
background = np.load('background.npy').astype(np.int16)
 
date = datetime.datetime.now()
session_name = date.strftime("%d-%m-%Y_%H-%M-%S")
os.makedirs(f"data/sessions/{session_name}", exist_ok=True)
timeout_count = [0]

TH = np.array([60, 60, 60])        # пороги по каналам
MIN_FG_PIXELS = 200                 # сколько отклонившихся пикселей в строке = "объект есть"
GAP_ROWS = 3                       # столько подряд фоновых строк = объект закончился
MAX_ROWS = 400                    # предохранитель от бесконечного объекта
ALPHA = 0.01
H, W = 164, 16

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'
 
val_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])

class SmallNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                    # 164x18 -> 82x9

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                    # -> 41x4

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                    # -> 20x4, высоту НЕ трогаем

            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                 # -> 1x1, любой вход
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)

def read_line():
    strings = []
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for _ in range(12):
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            timeout_count[0] +=1
            raise
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)
        strings.extend(bright)
    rgb = np.array(strings, dtype=np.uint8).reshape(-1, 3)
    return rgb[X0:X1]
 
 
def row_has_object(row):
    """True, если строка заметно отличается от фона."""
    diff = np.abs(row.astype(np.int16) - background)
    fg = (diff > TH).any(axis=1)  
    #print(fg.sum())                      
    return fg.sum() >= MIN_FG_PIXELS
 
 
def find_obj(image, session_name = None, need_save = True, threshold=(25, 25, 25), obj_counter=[0]):
    """image — собранный объект (строки, ширина, 3). Выделяет и сохраняет объекты."""
    r_th, g_th, b_th = threshold
    diff = np.abs(image.astype(np.int16) - background)
    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
 
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    MIN_AREA = 1000
    found = 0
    objects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        crop = image[y:y+h, x:x+w]
        if need_save:
            fname = f"data/sessions/{session_name}/object_{obj_counter[0]}.png"
            cv2.imwrite(fname, crop[:, :, ::-1])
        objects.append(crop[:, :, ::-1])       
        obj_counter[0] += 1
        found += 1
    if found:
        print("сохранено объектов:", found)
    return objects

def capture_loop():
    try:
        collecting = False
        obj_rows = []
        gap = 0
        t_start = None
        while True:
            try:
                row = read_line()
            except socket.timeout:
                continue

            if row_has_object(row):
                if not collecting:
                    t_start = time.perf_counter()
                obj_rows.append(row); collecting = True; gap = 0
            else:
                if collecting:
                    gap += 1; obj_rows.append(row)
                    if gap >= GAP_ROWS:
                        t_before_stack = time.perf_counter()
                        image = np.stack(obj_rows)
                        t_assembled = time.perf_counter()

                        print(f"[capture] сбор: {(t_assembled - t_start)*1000:.1f} мс, "
                              f"строк: {len(obj_rows)}, "
                              f"мс/строку: {(t_assembled - t_start)*1000/len(obj_rows):.1f}"
                              f"Количество таймаутов: {timeout_count}")
                            

                        obj_queue.put(image)
                        obj_rows = []; collecting = False; gap = 0
    except Exception:
        import traceback
        print('Основной цикл захвата упал')
        traceback.print_exc()

def inference_loop(rknn_work = False):
    try:
        model = SmallNet(n_classes=2)
        model.load_state_dict(torch.load('models/model.pth', map_location='cpu'))
        while True:
            image = obj_queue.get()
            t0 = time.perf_counter()

            objects = find_obj(image)
            t1 = time.perf_counter()

            for obj in objects:
                if rknn_work:
                    cls = inference_rknn(imag=val_transform(obj))
                else:
                    cls = eval(model=model, image=val_transform(obj), device=device)
            t2 = time.perf_counter()
            print(f'Класс объекта: {cls}')
            print(f"[infer] выделение (find_obj): {(t1-t0)*1000:.6f} мс, "
                  f"нейросеть ({len(objects)} об.): {(t2-t1)*1000:.6f} мс, "
                  f"итого цикл: {(t2-t0)*1000:.6f} мс")
            obj_queue.task_done()
    except Exception:
        import traceback
        print('Inference loop упал')
        traceback.print_exc()

t1 = threading.Thread(target=capture_loop, daemon=True)
t2 = threading.Thread(target=inference_loop, daemon=True)

t1.start()
t2.start()

t1.join()
t2.join()