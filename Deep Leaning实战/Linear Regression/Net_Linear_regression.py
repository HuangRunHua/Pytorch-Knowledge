import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from time import perf_counter, time

"""
本程序使用人工神经元实现线性回归
"""

class LR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Linear第一个参数表示输入数据的维度，第二个参数表示输出数据的维度
        # Linear为预设的线性神经网络模块
        self.linear = nn.Linear(1,1)

    # 前向传播函数（必须实现）
    # 利用当前参数w与输入数据来得到输出结果的预测，用于下一步计算损失函数
    def forward(self, x):
        out = self.linear(x)
        return out

def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        # 将inputs输入到神经网络模型中，此时调用forward函数
        output = model(inputs)
        # 调用Pytorch预设的损失函数来计算损失
        loss = criterion(output, target)
        # 将模型的参数梯度初始化为0
        optimizer.zero_grad()
        # 求解损失函数对w的梯度向量
        loss.backward()
        # 权值更新过程
        optimizer.step()
        if epoch % 80 == 0:
            draw(output,loss)
    return model, loss

def draw(output, loss):
    # 清空图像画布
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    # 打印loss值
    plt.text(0.5,0,'loss=%s' % (loss.item()), fontdict={'size':20, 'color':'red'})
    plt.pause(0.005)

# unsqueeze的作用和Produce_X作用一样
x = torch.unsqueeze(torch.linspace(-3,3,100000), dim=1)
# 实际输出
y = x + 1.2*torch.rand(x.size())

LR_model = LR()
inputs = x
target = y

# Pytorch预设的损失函数
criterion = nn.MSELoss()
# SGD第一个参数为需要优化的神经网络模型参数
# SGD为随机梯度下降函数
optimizer = optim.SGD(LR_model.parameters(), lr=1e-4)

start  = perf_counter()
LR_model, loss = train(LR_model, criterion, optimizer, 10000)
finish = perf_counter()
train_time = finish - start

print("Total Train Time: %s" % train_time)
print("final loss:", loss.item())

# 获取回归方程的参数
[w,b] = LR_model.parameters()
print(w.item())
print(b.item())
"""
输出结果:
0.9949362874031067
0.40299367904663086
"""