from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

"""
本程序利用RNN实现在已知sin值的条件下去预测cos的值
"""

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
input_x = np.sin(steps)
target_y = np.cos(steps)
plt.plot(steps, input_x, 'b-', label='input: sin')
plt.plot(steps, target_y, 'r-', label='target: cos')
plt.legend(loc='best')
plt.show()