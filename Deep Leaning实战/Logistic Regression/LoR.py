import torch
import matplotlib.pyplot as plt
from torch import nn, optim

"""
本代码实现两个集合的分类问题

集合特征个数为2, 因此输入数据为2维矩阵
模型的输出为两个类别的所属的概率, 为2维矩阵
"""

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 定义线性函数，参数为随机数
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        # 给线性函数赋值，计算给定输入x时候的输出
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

def draw(output):
    plt.cla()
    # max返回每一行中的最大值与对应的索引序号
    # output值为每一个数据所代表的类(通过概率来判断所属的类别)
    # output = [0,0, ... , 1,1,1]
    output = torch.max((output), 1)[1]
    pred_y = output.data.numpy().squeeze()
    target_y = y.numpy()
    plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')
    accuracy = sum(pred_y == target_y)/1000.0
    plt.text(1.5, -4, 'Accuracy=%s' % (accuracy), fontdict={'size': 20, 'color': 'red'})
    plt.pause(0.1)

def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        output = model.forward(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            draw(output)

cluster = torch.ones(500, 2)

"""
使用normal函数生成期望分别为4与-4, 标准差为2的随机数
每一批数据均为500个, 为2维数据
"""
data0 = torch.normal(4*cluster, 2)
data1 = torch.normal(-4*cluster, 2)

"""
生成两个数据集对应的标签
"""
label0 = torch.zeros(500)
label1 = torch.ones(500)

# 使用cat将两组数据整合在一起
x = torch.cat((data0, data1), ).type(torch.FloatTensor)
y = torch.cat((label0, label1), ).type(torch.LongTensor)
print(y)

# plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=10, lw=0, cmap='RdYlGn')
# plt.show()

net = Net()
inputs = x
target = y

"""
优化器选择随机梯度下降SGD
损失函数选择交叉熵函数
"""
optimizer = optim.SGD(net.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

train(net, criterion, optimizer, 1000)