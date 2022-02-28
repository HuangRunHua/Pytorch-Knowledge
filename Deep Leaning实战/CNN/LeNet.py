import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

"""
本程序使用CNN神经网络实现手写字体识别(MINST)
LeNet网络结构如下所示:
┌────────────────────────────────────────────────────────────────────────────────────────────┐
|               Conv                Pool                Conv               Pool              |
| Input(28x28)────────▶C1(6x24x24)────────▶S2(6x12x12)────────▶C3(16x8x8)────────▶S4(16x4x4) |  
|                                                                                  | Linear  |
|                                                                                  | Linear  |
|                                           Linear         Linear                  | Linear  |
|                               Output(10)◀────────F6(84)◀────────F5(120)◀─────────┘         |
└────────────────────────────────────────────────────────────────────────────────────────────┘  
 MINST问题为分类问题, 因此神经网络的输出为该图片中数字属于0-9共10种情况的概率。
"""

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
        原先x.size() = 16x4x4 维度=16, 现要转成1维度即x.size() = 1x(16x4x4)
        x.view(-1,given_number) <==> x.size() = given_number * x的维度
        由于given_number = 16x4x4, 因此x的维度 = 1
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
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opitimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                running_loss = 0.0
    print('Finished Training')


"""
保存与加载模型的方法:
方法一:
    - torch.save(lenet, 'model.pkl') 保存整个模型
    - lenet = torch.load('model.pkl) 加载模型
方法二:
    - torch.save(lenet.state_dict(), 'model.pkl') 保存模型参数
    - torch.load_state_dict(torch.load('model.pkl')) 加载模型参数
"""
def save_param(model, path):
    torch.save(model.state_dict(), path)

def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

"""
神经网络的测试部分, 由于神经网络的输出为0-9与图片标签相同
可以采用输出与标签做差的方式求取预测的准确率
"""
def test(testloader, model):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = model.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy on the test set: %d %%' %(100*correct/total))


"""
程序开始的位置
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

# 初始化神经网络
lenet = LeNet()

"""
┌────────────────────────────────────┐
|             CNN Training           |
└────────────────────────────────────┘
"""

"""
batch_size表示一次性加载数据量
shuffle=True表示打乱数据集
num_workers=2表示使用两个子进程加载数据
(注意: 当前m1芯片的MacBook Air无法使用GPU因此设定num_workers=0)
"""
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

load_param(lenet, 'model.pkl')
train(lenet, criterion, optimizer, epochs=2)
save_param(lenet, 'model.pkl')

"""
┌────────────────────────────────────┐
|             CNN Testing            |
└────────────────────────────────────┘
"""
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)
test(testloader, lenet)

