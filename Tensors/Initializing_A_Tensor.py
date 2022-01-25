import torch
import numpy as np

###############################################################
# [1] 直接来自数据。
# 张量可以直接从数据中创建。 数据类型是自动推断的。
###############################################################
data = [[1, 2],
        [3, 4]]
x_data = torch.tensor(data)
"""
>>> tensor([[1, 2],
            [3, 4]])
"""

###############################################################
# [2] 来自 NumPy 数组
# 张量可以从 NumPy 数组创建（反之亦然 - 请参阅 Bridge with NumPy）。
###############################################################
np_array = np.array(data)
"""
>>> [[1 2]
     [3 4]]
"""
x_np = torch.from_numpy(np_array)

###############################################################
# [3] 从另一个张量：
# 新张量保留参数张量的属性（形状、数据类型），
# 除非显式覆盖。
###############################################################

# retains the properties of x_data
x_ones = torch.ones_like(x_data) 
print(f"Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor: \n {x_rand} \n")
"""
>>> Ones Tensor: 
    tensor([[1, 1],
            [1, 1]]) 

>>> Random Tensor: 
    tensor([[0.4423, 0.3299],
            [0.9035, 0.9328]]) 
"""

###############################################################
# [4] 使用随机或恒定值：
# shape 是张量维度的元组。 在下面的函数中，
# 它决定了输出张量的维度。
###############################################################
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
"""
>>> Random Tensor: 
    tensor([[0.5289, 0.1512, 0.5405],
            [0.5642, 0.8397, 0.8233]]) 

>>> Ones Tensor: 
    tensor([[1., 1., 1.],
            [1., 1., 1.]]) 

>>> Zeros Tensor: 
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
"""