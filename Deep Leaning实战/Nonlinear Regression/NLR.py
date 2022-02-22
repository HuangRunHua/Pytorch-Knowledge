import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

"""
使用Pytorch利用神经网络实现非线性回归
非线性函数选用三次方程

在无激活函数的情况下, 神经网络相当于多个线性模型进行叠加
激活函数的出现使得网络可以拟合非线性的函数

每一个神经元最终的输出在激活函数内进行, 多层神经元组成人工神经网络

torch.nn.Linear(in_features, out_features):
    - in_features: 每一个输入样本的大小
    - out_features: 每一个输出样本的大小

本案例激活函数选取`torch.nn.functional.relu()`
"""

class Net(nn.Module):
    # num_hidden为隐含层节点数
    # input_feature为输入的维度
    def __init__(self, input_feature, num_hidden, outputs) -> None:
        super().__init__()
        # 定义神经网络的隐含层
        self.hidden = nn.Linear(input_feature, num_hidden)
        # 定义神经网络的输出层
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        # 调用激活函数
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        # 将inputs输入到神经网络模型中，此时调用forward函数
        output = model.forward(inputs)
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

# 指定输入数据, 其维度为1维
x = torch.unsqueeze(torch.linspace(-3, 3, 10000), dim=1)
# 给输入添加噪声来模拟实际情况, 输出数据维度也为1维
# 因此神经网络中`self.out`最终维度为1
y = x.pow(3) + 0.3*torch.rand(x.size())

# plt.scatter(x.numpy(), y.numpy(), s=0.01)
# plt.show()

net = Net(input_feature=1, num_hidden=20, outputs=1)
inputs = x
target = y

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

net, loss = train(net, criterion, optimizer, 10000)
print("final loss:", loss.item())