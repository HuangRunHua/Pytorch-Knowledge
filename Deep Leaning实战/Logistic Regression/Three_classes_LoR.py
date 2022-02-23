import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim

"""
本代码实现三个集合的分类问题

本代码具备输入层, 隐藏层与输出层, 实际上隐藏层非必需层
"""

class Net(nn.Module):
    def __init__(self, input_feature, num_hidden, outputs) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        x = F.softmax(x)
        return x

def draw(output):
    plt.cla()
    output = torch.max((output), 1)[1]
    pred_y = output.data.numpy().squeeze()
    target_y = y.numpy()
    plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')
    accuracy = sum(pred_y == target_y)/1500.0
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
data1 = torch.normal(-4*cluster, 1)
data2 = torch.normal(-8*cluster, 1)

"""
生成两个数据集对应的标签
"""
label0 = torch.zeros(500)
label1 = torch.ones(500)
label2 = label1*2

# 使用cat将两组数据整合在一起
x = torch.cat((data0, data1, data2), ).type(torch.FloatTensor)
y = torch.cat((label0, label1, label2), ).type(torch.LongTensor)

net = Net(input_feature=2, num_hidden=20, outputs=3)
inputs = x
target = y

optimizer = optim.SGD(net.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

train(net, criterion, optimizer, 10000)

