from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim

"""
本程序使用CNN神经网络实现手写字体识别(MINST)
 ---------------------------------------------------------------------------------------
|            卷积               池化                卷积               池化                |
|输入(28x28)------->C1(6x24x24)------->S2(6x12x12)------->C3(16x8x8)------->S4(16x4x4)   |  
|                                                                               |全连    |
|                                                                               |全连    |
|                                         高斯连          全连层                  |全连    |
|                               输出(10)<--------F6(84)<--------F5(120)<---------        |
 ---------------------------------------------------------------------------------------  
"""

transform = transforms.Compose([
    # 将加载的数据转为Tensor对象
    transforms.ToTensor(),
    # 将数据进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

"""
'data'代表数据集下载存储文件夹
train=True代表加载训练集
download=True代表自动下载数据集
"""
trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
testset = datasets.MNIST('data', train=False, download=True, transform=transform)

class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        """
        C1为卷积层, 1表示输入为一张灰度图, 6表示输出6张特征图
          (5,5)表示5x5卷积核过滤器, 可简化为5
        fc1~fc3为全连接层
        """
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.c3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        卷积后使用relu增强网络的非线性拟合能力
        使用max_pool2d对c1的特征图进行池化, 池化核大小为2x2,简写为2
        """
        x = F.max_pool2d(F.relu(self.c1(x)),2)
        x = F.max_pool2d(F.relu(self.c3(x)),2)
        """
        view将x的形状转化成1维向量
        """
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """
        在全连接层之前的数据为16x4x4, 即16张4x4大小的特征图
        pytorch只接收同时处理多张图片, 假设同时输入a张图片, 则x.size()将会变成ax16x4x4, 
        在输入到全连接层的时候x.size()需要为16x4x4
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(model, criterion, opitimizer, epochs=1):
    for epoch in range(epochs):
        running_loss = 0.0
        """
        enumerate(trainloader, 0)表示从第0项开始对trainloader中的数据进行枚举
        返回i为序号, data为数据,包含训练数据与标签
        """
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opitimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                running_loss = 0.0
    print(i, data)
    print('Finished Training')


lenet = LeNet()

"""
batch_size表示一次性加载数据量
shuffle=True表示打乱数据集
num_workers=2表示使用两个子进程加载数据
(注意: 当前m1芯片的MacBook Air无法使用GPU因此设定num_workers=0)
"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train(lenet, criterion, optimizer, epochs=2)

