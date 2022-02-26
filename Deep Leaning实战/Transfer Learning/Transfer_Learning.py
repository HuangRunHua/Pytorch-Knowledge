import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


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

# 数据集文件夹路径
data_directory = 'data_set'

trainset = datasets.ImageFolder(os.path.join(data_directory, 'train'), data_transforms['train'])
testset = datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True, num_workers=0)

inputs, classes = next(iter(trainloader))
imshow(torchvision.utils.make_grid(inputs))
