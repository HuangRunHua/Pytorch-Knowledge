import torch
import numpy as np

# 张量运算
# 超过 100 种张量运算，包括算术、线性代数、矩阵操作（转置、索引、
# slicing)、sampling 等在这里进行了全面的描述。
# 这些操作中的每一个都可以在 GPU 上运行（通常比在 CPU 上的速度更高）。
# 如果您使用的是 Colab，请转到运行时 > 更改运行时类型 > GPU 来分配 GPU。
# 默认情况下，张量是在 CPU 上创建的。 我们需要使用“.to”方法（在检查 GPU 可用性之后）将张量显式移动到 GPU。
# 请记住，跨设备复制大张量在时间和内存方面可能会很昂贵！

tensor = torch.ones(4, 4)
tensor[:,1] = 0

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