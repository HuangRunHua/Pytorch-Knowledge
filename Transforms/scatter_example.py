import torch
"""
本程序模拟实现.scatter_(dim, index, src)的功能
    - dim 就是在哪个维度进行操作
    - index 是输入的索引
    - src 就是输入的向量，也就是 input。 
    - 函数返回一个 Tensor

具体计算方法如下:
    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    elf[i][j][index[i][j][k]] = src[i][j][k]   # if dim == 2
"""

src = torch.rand(2,5)
"""
>>> src = tensor([[0.4028, 0.5413, 0.9855, 0.4965, 0.5300],
                  [0.4541, 0.2645, 0.2957, 0.9252, 0.0092]])
"""

b = torch.zeros(3,5).scatter_(
    dim=0,
    index=torch.LongTensor([[0,1,2,0,0], 
                            [2,0,0,1,2]]),
    src=src
)
"""
>>> b = tensor([[0.4028, 0.2645, 0.2957, 0.4965, 0.5300],
                [0.0000, 0.5413, 0.0000, 0.9252, 0.0000],
                [0.4541, 0.0000, 0.9855, 0.0000, 0.0092]])
"""

c = torch.zeros(3,5).scatter_(
    dim=1,
    index=torch.LongTensor([[0,1,2,0,0], 
                            [2,0,0,1,2]]),
    src=src
)
"""
>>> c = tensor([[0.5300, 0.5413, 0.9855, 0.0000, 0.0000],
                [0.2957, 0.9252, 0.0092, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
"""

d = torch.zeros(3,5).scatter_(
    dim=0,
    index=torch.LongTensor([[0,1,2,0,0], 
                            [2,0,0,1,2]]),
    value=1
)
"""
>>> d = tensor([[1., 1., 1., 1., 1.],
                [0., 1., 0., 1., 0.],
                [1., 0., 1., 0., 1.]])
"""


