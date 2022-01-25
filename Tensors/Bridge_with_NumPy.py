import torch
import numpy as np

# 和 NumPy 的连接
# CPU 和 NumPy 数组上的张量可以共享它们的底层内存位置，改变一个会改变另一个。

# Tensor 转 Numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
"""
>>> t: tensor([1., 1., 1., 1., 1.])
>>> n: [1. 1. 1. 1. 1.]
"""

# 改变tensor的值将会影响numpy
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
"""
>>> t: tensor([2., 2., 2., 2., 2.])
>>> n: [2. 2. 2. 2. 2.]
"""

# Numpy数组到Tensor的转换
n = np.ones(5)
t = torch.from_numpy(n)

# np.add()操作将会改变n的值，np没有.add_()函数
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

"""
>>> t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
>>> n: [2. 2. 2. 2. 2.]
"""