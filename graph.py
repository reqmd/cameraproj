import socket
import numpy as np
import matplotlib.pyplot as plt

CAMERA_IP   = "192.168.2.101"   
CAMERA_PORT = 5001   
WIDTH = 2716 
cmd = bytes.fromhex("01 00 03 00 00 00 6A 00 00 1F D4 8e 06")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

x = np.arange(WIDTH) 

plt.ion()
fig, ax = plt.subplots()
line_r, = ax.plot(x, np.zeros(WIDTH), 'r-', label='R')
line_g, = ax.plot(x, np.zeros(WIDTH), 'g-', label='G')
line_b, = ax.plot(x, np.zeros(WIDTH), 'b-', label='B')
ax.set_ylim(0, 255)
ax.set_xlim(0, WIDTH)
ax.set_xlabel('Линия пикселей')
ax.set_ylabel('Яркость')
ax.legend()

def update_hist(rgb):
    """rgb — массив (пиксели, 3) одной строки."""
    line_r.set_ydata(rgb[:, 0]) 
    line_g.set_ydata(rgb[:, 1])
    line_b.set_ydata(rgb[:, 2])
    fig.canvas.draw()
    fig.canvas.flush_events()

while True:
    strings = []
    sock.settimeout(0.1)
    sock.sendto(cmd, (CAMERA_IP, CAMERA_PORT))
    for i in range(12):
        data, addr = sock.recvfrom(65535)      # сеть
        vals = np.frombuffer(data[12:-2], dtype='>u2')
        bright = (vals / 64).clip(0, 255).astype(np.uint8)   # вычисления
        strings.extend(bright)
    rgb = np.array(strings, dtype=np.uint8).reshape(-1, 3)
    update_hist(rgb)
sock.close()
