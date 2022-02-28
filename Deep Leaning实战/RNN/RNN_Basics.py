import torch
from torch import nn

"""
本程序介绍RNN的相关知识点
输入: (x_0, x_1, ..., x_t)

循环神经网络展开示意图:
 --------------------------------------------------------------------------------
|           y_t                 y_0                            y_t                |
|            ^                   ^                              ^                 |
|    --------|                   |                              |                 |
|   |  --------------     --------------   h_0           --------------   h_t     |
|   | | hidden layer | = | hidden layer |------> ... -->| hidden layer |------>   |
|   |  --------------     --------------                 --------------           |
|   |        ^                   ^                              ^                 |
|    ------->|                   |                              |                 |
|           x_t                 x_0                            x_t                |
|                                                                                 |
 --------------------------------------------------------------------------------

RNN的数学本质:
h_t = tanh(w_ih * x_t + b_ih + w_hh * h_{t-1} + b_hh)
"""

# 初始化RNN单元, 输入特征维度为5, 隐藏向量维度特征为7
rnn_cell = nn.RNNCell(input_size=5, hidden_size=7)
"""
>>> RNNCell(5, 7)
"""

"""
利用weigt_ih, weight_hh, bias_ih, bias_hh访问公式中w_ih, w_hh, b_ih与b_hh的参数值
"""
print(rnn_cell.weight_ih)
"""
>>> tensor([[-0.2274, -0.0873, -0.2169,  0.0259,  0.1503],
            [-0.0536, -0.2459, -0.2192,  0.3059,  0.1788],
            [ 0.3169,  0.3081, -0.2950,  0.3668,  0.2080],
            [ 0.2288, -0.1072, -0.3589,  0.2993, -0.1471],
            [-0.1201, -0.0550,  0.0778, -0.3691,  0.0984],
            [-0.2739, -0.2830,  0.0401,  0.2677, -0.1522],
            [ 0.3303, -0.3012,  0.2901,  0.3631, -0.0831]], requires_grad=True)
"""


"""
 -------------------------
|      初始化RNN Cell      |
 ------------------------- 
"""
inputs = torch.randn(1, 5)
hidden = torch.randn(1, 7)
user_def_rnn_cell = rnn_cell(inputs, hidden)
"""
注: 0.3787,..., -0.4262代表隐藏向量
>>> tensor([[ 0.3787, -0.1586,  0.6216,  0.4170,  0.4745,  0.1544, -0.4262]],
       grad_fn=<TanhBackward0>)
"""

"""
 --------------------
|      初始化RNN      |
 -------------------- 
"""
rnn = nn.RNN(input_size=5, hidden_size=7)

inputs_rnn = torch.randn(3, 2, 5)
hidden_rnn = torch.randn(1, 2, 7)
user_def_rnn = rnn(inputs_rnn, hidden_rnn)
"""
>>> (tensor([[[ 0.4635,  0.5563,  0.7083,  0.1609,  0.6778,  0.2823,  0.4436],
              [ 0.4959, -0.3169, -0.9003,  0.8274,  0.7308, -0.1930,  0.1689]],

              [[ 0.0439,  0.5051, -0.5075,  0.1898,  0.6566,  0.3137, -0.9038],
              [-0.0750, -0.5077, -0.2272,  0.4561, -0.5997,  0.0679, -0.0113]],

              [[ 0.4049,  0.5675,  0.5623, -0.0836,  0.6131,  0.7516, -0.6377],
              [-0.3210, -0.0963,  0.0514,  0.5177,  0.1650, -0.2522,  0.6773]]],
    grad_fn=<StackBackward0>), 
    tensor([[[ 0.4049,  0.5675,  0.5623, -0.0836,  0.6131,  0.7516, -0.6377],
             [-0.3210, -0.0963,  0.0514,  0.5177,  0.1650, -0.2522,  0.6773]]],
    grad_fn=<StackBackward0>))
"""

