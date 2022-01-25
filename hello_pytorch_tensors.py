################################################################################################################
# 张量是一种特殊的数据结构，与数组和矩阵非常相似。
# 在 PyTorch 中，我们使用张量来编码模型的输入和输出，以及模型的参数。
#
# 张量类似于 NumPy 的 ndarray，只是张量可以在 GPU 或其他硬件加速器上运行。
# 实际上，张量和 NumPy 数组通常可以共享相同的底层内存，
# 消除复制数据的需要（参见使用 NumPy 进行桥接）。
# 张量也针对自动微分进行了优化。
# 如果你熟悉 ndarrays，你就会对 Tensor API 熟悉。
################################################################################################################

import torch
import numpy as np

#########################################################################################################
# 初始化张量
# 张量可以通过多种方式初始化。
# [1] 直接来自数据。
# 张量可以直接从数据中创建。 数据类型是自动推断的。
#########################################################################################################
data = [[1, 2],
        [3, 4]]
x_data = torch.tensor(data)
"""
>>> tensor([[1, 2],
            [3, 4]])
"""

#########################################################################################################
# [2] 来自 NumPy 数组
# 张量可以从 NumPy 数组创建（反之亦然 - 请参阅 Bridge with NumPy）。
#########################################################################################################
np_array = np.array(data)
"""
>>> [[1 2]
     [3 4]]
"""
x_np = torch.from_numpy(np_array)

#########################################################################################################
# [3] 从另一个张量：
# 新张量保留参数张量的属性（形状、数据类型），
# 除非显式覆盖。
#########################################################################################################

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

#########################################################################################################
# [4] 使用随机或恒定值：
# shape 是张量维度的元组。 在下面的函数中，
# 它决定了输出张量的维度。
#########################################################################################################
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

# 张量运算
# 超过 100 种张量运算，包括算术、线性代数、矩阵操作（转置、索引、
# slicing)、sampling 等在这里进行了全面的描述。
# 这些操作中的每一个都可以在 GPU 上运行（通常比在 CPU 上的速度更高）。
# 如果您使用的是 Colab，请转到运行时 > 更改运行时类型 > GPU 来分配 GPU。
# 默认情况下，张量是在 CPU 上创建的。 我们需要使用“.to”方法（在检查 GPU 可用性之后）将张量显式移动到 GPU。
# 请记住，跨设备复制大张量在时间和内存方面可能会很昂贵！

# 如果可用，我们将张量移动到 GPU，然而Mac系统下不可用
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
# 修改第一列所有行的元素为0
tensor[:,1] = 0

# 连接张量您可以使用“torch.cat”沿给定维度连接一系列张量。
# 另见“torch.stack”，另一个与torch.cat略有不同的张量加入操作。
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# torch.stack的用法在于不改变数据的shape的情况下，增加一维度来达到扩充张量的效果
T1 = torch.tensor([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
T2 = torch.tensor([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])
# 新增加的维度在于拼接的个数，T1与T2共两个tensors叠加，因此最终生成的张量情况有三种，即(2x3x3), (3x2x3)与(3,3,2)
# dim的数可以看成2在[3,3]列表里的位置
# dim为0时为(2x3x3), 以此类推
print(torch.stack((T1,T2),dim=0))

# 算术运算
# 张量的矩阵乘法运算方法，三种方法
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 张量的元素乘积，三种方法
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 单元素张量
# 如果您有一个单元素张量，例如通过将张量的所有值聚合为一个值，您可以使用 item() 将其转换为 Python 数值：
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
"""
>>> 12.0 <class 'float'>
"""

# 就地操作 
# 将结果存储到操作数中的操作称为就地操作。 它们用 _ 后缀表示。 
# 例如：x.copy_(y)，x.t_()，会改变x。
print(tensor, "\n")
tensor.add_(5)
print(tensor)
"""
>>> tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])

>>> tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]])
"""

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

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

"""
>>> t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
>>> n: [2. 2. 2. 2. 2.]
"""