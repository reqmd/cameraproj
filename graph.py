import socket
import numpy as np
import matplotlib.pyplot as plt

CAMERA_IP   = "192.168.2.101"   
CAMERA_PORT = 5001    

def update_hist(rgb):
    """rgb — массив (пиксели, 3) одной строки."""
    hr, _ = np.histogram(rgb[:, 0], bins=256, range=(0, 256))
    hg, _ = np.histogram(rgb[:, 1], bins=256, range=(0, 256))
    hb, _ = np.histogram(rgb[:, 2], bins=256, range=(0, 256))

    line_r.set_ydata(hr)           # меняем только Y, X постоянный
    line_g.set_ydata(hg)
    line_b.set_ydata(hb)

    #ax.set_ylim(0, max(hr.max(), hg.max(), hb.max()) + 1)   # подстроить высоту
    ax.set_xlim(0, 2716)
    fig.canvas.draw()
    fig.canvas.flush_events()

# bytes.fromhex умеет пробелы — можно писать по-байтно для читаемости:
cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

y = np.arange(256) 

plt.ion()
fig, ax = plt.subplots()
line_r, = ax.plot(np.zeros(2716), y, 'r-', label='R')
line_g, = ax.plot(np.zeros(2716), y, 'g-', label='G')
line_b, = ax.plot(np.zeros(2716), y, 'b-', label='B')
ax.set_ylim(0, 255)
ax.set_xlabel('Линия пикселей')
ax.set_ylabel('Яркость')
ax.legend()

while True:
    strings = []
    sock.settimeout(0.01)
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for i in range(12):
        data, addr = sock.recvfrom(65535)      # сеть
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)   # вычисления
        #print(bright)
        strings.extend(bright)
    rgb = np.array(strings, dtype=np.uint8).reshape(-1, 3)
    #print(bright)
    update_hist(rgb)
sock.close()
