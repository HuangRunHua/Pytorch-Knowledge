import torch
import numpy as np

# 张量的属性
# 张量属性描述了它们的形状、数据类型和存储它们的设备。
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
"""
>>> Shape of tensor: torch.Size([3, 4])
>>> Datatype of tensor: torch.float32
>>> Device tensor is stored on: cpu
"""