import torch

"""
本代码给出torch.sigmoid()函数的使用示例
"""

a = torch.randn(2)
"""
>>> tensor([ 1.9000, -0.3532])
"""

a = torch.sigmoid(a)
"""
>>> tensor([0.8699, 0.4126])
"""