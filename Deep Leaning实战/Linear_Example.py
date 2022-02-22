import torch
import torch.nn as nn

"""
本程序分析torch中Linear函数的使用方法
"""
x = torch.tensor([[1.0, -1.0],
                  [0.0,  1.0],
                  [0.0,  0.0]])

# in_features = 2
in_features = x.shape[1]
out_features = 2

"""
in_features指定输入数据的维度
    - x[0], x[1] and x[3], each of size 2
out_features指定最终的输出数据y的大小
    - (batch size, out_features) = (3, 2).
Linear的参数均代表矩阵的列数
"""
m = nn.Linear(in_features, out_features)

"""
使用`.weight`与`.bias`获取线性叠加的参数值与偏移量
`.weight`与`.bias`的值均为随机生成
"""
weight = m.weight
bias = m.bias

"""
使用`y = m(x)`获取最终的输出结果
由于`.weight`与`.bias`的值均为随机生成, 因此y也是随机生成的

上述y的计算等价于如下式子:
`y = x.matmul(m.weight.t()) + m.bias`
即为线性的计算公式: `y = x*W^T + b`
"""
y = m(x)