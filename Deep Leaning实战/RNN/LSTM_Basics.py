import torch
from torch import nn

"""
本程序介绍RNN改进版SLTM的相关知识点
输入: (x_0, x_1, ..., x_t)

LSTM结构如下:
┌───────────────────────────────────────────────────┐
|     c_{t-1}    ┌──────────────────┐      c_t      |
|   ───────────▶ |                  | ───────────▶  |
|     h_{t-1}    |   LSTM NetWork   |     h_t       |
|   ───────────▶ |                  | ───────────▶  |
|                └──────────────────┘               |
|                          ▲                        |
|                          |                        |   
|                          |                        |
|                         x_t                       |
└───────────────────────────────────────────────────┘
"""

lstm_cell = nn.LSTMCell(input_size=5, hidden_size=7)
"""
>>> LSTMCell(5, 7)
"""

inputs = torch.randn(1, 5)
h0 = torch.randn(1, 7)
c0 = torch.randn(1, 7)

user_def_lstm_cell = lstm_cell(inputs, (h0, c0))
"""
>>> user_def_lstm_cell
>>> tensor([[ 0.1523,  0.1727, -0.3617,  0.0911, -0.0683, -0.0848, -0.4115]],
       grad_fn=<MulBackward0>), 
    tensor([[ 0.5382,  0.6192, -1.1791,  0.2368, -0.1694, -0.2025, -0.9808]],
       grad_fn=<AddBackward0>)
"""