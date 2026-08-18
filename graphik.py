import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 4, 9, 16, 25])

plt.plot(a, b)
plt.show()