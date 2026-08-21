import multiprocessing as mp
import numpy as np
import socket
import cv2
import datetime
import os
import time
import yaml


# ============================ ПРОЦЕСС КЛАССИФИКАЦИИ ============================
# Живёт отдельно: свой интерпретатор, свой GIL, свой экземпляр RKNN.
# RKNN создаётся ОДИН раз здесь, а не на каждый объект.
def inference_process(crop_queue, rknn_path, classes={1: 'Скрепка', 0: 'Зерно'}):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # прогрев — первый вызов всегда медленный
    warm = np.zeros((1, 164, 16, 3), dtype=np.uint8)
    for _ in range(5):
        rknn.inference(inputs=[warm])

    print('[infer] RKNN готов, жду объекты')
    while True:
        item = crop_queue.get()
        if item is None:          # сигнал завершения
            break
        crop, meta = item         # crop уже (1,164,16,3) uint8

        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[crop])
        #print(outputs[0])
        pred = int(np.argmax(outputs[0]))
        #print(pred)
        dt = (time.perf_counter() - t0) * 1000

        print(f"[infer] класс: {classes[pred]}  (инференс {dt:.3f} мс, "
              f"объект пришёл из строк={meta.get('rows')})")

    rknn.release()


# ============================ ЗАХВАТ (ОСНОВНОЙ ПРОЦЕСС) ========================
CAMERA_IP   = "192.168.2.101"
CAMERA_PORT = 5001
WIDTH = 2716
X0, X1 = 175, WIDTH - 1000

cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")

TH = np.array([60, 60, 60])
MIN_FG_PIXELS = 50
GAP_ROWS = 3
MAX_ROWS = 400
ALPHA = 0.01
H, W = 164, 16

timeout_count = [0]


def read_line(sock):
    strings = []
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for _ in range(12):
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            timeout_count[0] += 1
            raise
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)
        strings.extend(bright)
    rgb = np.array(strings, dtype=np.uint8).reshape(-1, 3)
    return rgb[X0:X1]


def collect_background(sock, n_lines=20):
    print(f"калибровка фона: собираю {n_lines} строк, лента должна быть пустой...")
    rows = []
    while len(rows) < n_lines:
        try:
            rows.append(read_line(sock))
        except socket.timeout:
            continue
    bg = np.median(np.stack(rows), axis=0).astype(np.int16)
    print("фон собран, форма:", bg.shape)
    return bg


def row_has_object(row, background):
    diff = np.abs(row.astype(np.int16) - background)
    fg = (diff > TH).any(axis=1)
    return fg.sum() >= MIN_FG_PIXELS


def find_obj(image, background, session_name, need_save=True,
             threshold=(25, 25, 25), obj_counter=[0]):
    r_th, g_th, b_th = threshold
    diff = np.abs(image.astype(np.int16) - background)
    mask = ((diff[:, :, 0] > r_th) |
            (diff[:, :, 1] > g_th) |
            (diff[:, :, 2] > b_th)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    MIN_AREA = 1000
    objects = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        crop = image[y:y+h, x:x+w][:, :, ::-1]     # RGB
        if need_save:
            fname = f"data/sessions/{session_name}/object_{obj_counter[0]}.png"
            cv2.imwrite(fname, crop)
            obj_counter[0] += 1
        objects.append(crop)
    return objects


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.1)

    background = collect_background(sock, 20)
    print('Фон собран!')

    session_name = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.makedirs(f"data/sessions/{session_name}", exist_ok=True)

    with open('models.yaml', 'r') as f:
        rknn_path = yaml.safe_load(f)['rknn']

    # запускаем процесс классификации
    crop_queue = mp.Queue(maxsize=20)
    p = mp.Process(target=inference_process, args=(crop_queue, rknn_path), daemon=True)
    p.start()

    # автомат сборки объектов
    collecting = False
    obj_rows = []
    gap = 0
    t_start = None

    while True:
        try:
            row = read_line(sock)
        except socket.timeout:
            continue

        if row_has_object(row, background):
            if not collecting:
                t_start = time.perf_counter()
            obj_rows.append(row); collecting = True; gap = 0
            if len(obj_rows) > MAX_ROWS:
                obj_rows = []; collecting = False; gap = 0
        else:
            # адаптивный фон — только по пустым строкам
            background = ((1 - ALPHA) * background + ALPHA * row).astype(np.int16)
            if collecting:
                gap += 1; obj_rows.append(row)
                if gap >= GAP_ROWS:
                    image = np.stack(obj_rows)
                    n_rows = len(obj_rows)
                    obj_rows = []; collecting = False; gap = 0

                    objects = find_obj(image, background, session_name)
                    for obj in objects:
                        crop = cv2.resize(obj, (16, 164))          # (W, H)
                        crop = np.expand_dims(crop, axis=0)         # (1,164,16,3)
                        try:
                            crop_queue.put((crop, {'rows': n_rows}), block=False)
                        except Exception:
                            pass    # очередь полна — пропускаем, чтобы не тормозить приём


if __name__ == '__main__':
    main()