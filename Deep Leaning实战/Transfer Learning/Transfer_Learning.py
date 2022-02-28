import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms, models


"""
本程序利用AlexNet模型进行迁移学习, 实现柴犬与猫猫的图像分类器
"""


"""
随机展示训练的样本
"""
def imshow(inputs):
    inputs = inputs / 2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.show()

"""
定义迁移学习的训练函数
"""
def train(model, criterion, optimizer, epochs=1):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch+1, i+1, running_loss/100))
                running_loss = 0.0
    print('Finished Training')


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

def save_param(model, path):
    torch.save(model.state_dict(), path)

def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

"""
数据预处理部分:
    .scale: 将图片自适应缩小或放大
    .centercrop: 居中剪裁图片
    .randomhorizontalflip: 随机水平翻转
    .totensor: 将图片转化为tensor格式
    .normalize: 归一化

其他常用的操作:
    .RandomCrop: 随机切割
    .ToPILImage: 将tensor对象转为PIL图像
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(230),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

"""
数据加载部分

训练与测试数据放置于`data_set`文件夹内
    注意: `data_set`文件夹需要放置在路径os.getcwd()下
"""
# 数据集文件夹路径
data_directory = 'data_set'
trainset = datasets.ImageFolder(os.path.join(data_directory, 'train'), data_transforms['train'])
testset = datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=0)

inputs, classes = next(iter(trainloader))
imshow(torchvision.utils.make_grid(inputs))

"""
使用经典的AlexNet模型
pretrained=True代表加载经过了ImageNet数据集训练后的模型参数
print(alexnet)可以直接打印模型的结构

AlexNet模型结构:
    1. features模块: 负责提取特征
    2. classifier模块: 负责实现分类, 以全连接层为主

本实验为二分问题, 因此需要重新定义AlexNet的classifier模块,
将最后一层的输出改为2
"""
alexnet = models.alexnet(pretrained=True)

for param in alexnet.parameters():
    # 限制参数的更新, 保证只更新classifier的参数
    param.requires_grad = False

alexnet.classifier=nn.Sequential(
    nn.Dropout(),
    nn.Linear(256*6*6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 2),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.classifier.parameters(), lr=0.001, momentum=0.9)

"""
┌─────────────────────────────────────────────────┐
|             AlexNet Train and Test              |
└─────────────────────────────────────────────────┘
"""
load_param(alexnet, 'tl_model.pkl')
train(alexnet, criterion, optimizer, epochs=2)
save_param(alexnet, 'tl_model.pkl')
test(testloader, alexnet)
