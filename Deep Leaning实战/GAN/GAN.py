import os
from sklearn.utils import shuffle
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

"""
本程序使用Generative Adversarial Network(GAN)实现柴犬头像的生成

Generator解析: 
    1. 一个神经网络或者一个函数
    2. 向生成器中输人一个向量, 就可以输出一些东西。
    3. 输人一个向量, 生成器便会生成一张图片。通常, 输人向量的每一个维度都会对应图片的某一种特征。
    ┌─┐         ┌───────────┐
    | |         |           |         ┌─────┐
    | |────────▶| Generator |────────▶| IMG |
    | |         |           |         └─────┘
    └─┘         └───────────┘        
    向量

Discriminator解析:
    1. 用于训练生成器
    2. 一个神经网络或者一个函数
    3. 输出为标量, 0~1, 接近1表示图片或数据越真实
                    ┌───────────┐
    ┌─────┐         |           |         
    | IMG |────────▶|  Discrim  |────────▶标量
    └─────┘         |           |
                    └───────────┘

生成对抗网络训练逻辑:
        ┌──────────┐          ┌──────────┐           ┌──────────┐
        |          |          |          |           |          |
        | Gene 1.0 |─────────▶| Gene 2.0 |──────────▶| Gene 3.0 |
        |          |          |          |           |          |
        └──────────┘          └──────────┘           └──────────┘
             |                      |                      |
             |                      |                      |
             ▼                      ▼                      ▼
    ┌───┐───┐───┐───┐───┐  ┌───┐───┐───┐───┐───┐  ┌───┐───┐───┐───┐───┐
    |   |   |   |   |   |  |   |   |   |   |   |  |   |   |   |   |   |
    └───┘───┘───┘───┘───┘  └───┘───┘───┘───┘───┘  └───┘───┘───┘───┘───┘
          生成的图片               生成的图片              生成的图片
             |                      |                      |
             |                      |                      |
             ▼                      ▼                      ▼
        ┌──────────┐          ┌──────────┐           ┌──────────┐
        |          |          |          |           |          |
        | Disc 1.0 |─────────▶| Disc 2.0 |──────────▶| Disc 3.0 |
        |          |          |          |           |          |
        └──────────┘          └──────────┘           └──────────┘
             ▲                      ▲                      ▲
             |                      |                      |
             |                      |                      |
             |             ┌───┐───┐───┐───┐───┐           |
             └─────────────|   |   |   |   |   |───────────┘
                           └───┘───┘───┘───┘───┘
                                  真实图片

GAN算法流程:
    1. 初始化生成器和鉴别器的参数
    2. 每次训练迭代中进行如下操作:
        a. 固定生成器, 升级鉴别器
        b. 固定鉴别器, 升级生成器

固定生成器升级鉴别器图解:
                                                                                   Self update
                                                                               ┌──────────────────┐
                                                                               |                  |
                                        True Pictures                          ▼                  |
┌───┐───┐───┐───┐───┐    Sample     ┌───┐───┐───┐───┐───┐               ┌─────────────┐           |
|   |   |   |   |   |──────────────▶|   |   |   |   |   |──────────────▶|             |           |
└───┘───┘───┘───┘───┘               └───┘───┘───┘───┘───┘               |   Discrim   |───────────┘
    True Data                       ┌───┐───┐───┐───┐───┐               |     3.0     | 
                                    |   |   |   |   |   |──────────────▶|             |
                                    └───┘───┘───┘───┘───┘               └─────────────┘
                                        Gene Pictures
                                              ▲
                                              |
                                              |
                ┌─┐┌─┐┌─┐                ┌──────────┐
                | || || |                |          |
                | || || |──────────────▶ |   Gene   |
                | || || |                |          |
                | || || |                └──────────┘
                └─┘└─┘└─┘                    Fixed 
              Random Vectors 

固定鉴别器升级生成器图解:
    ┌─┐         ┌───────────┐                         ┌───────────┐
    | |         |           |         ┌─────┐         |           |
    | |────────▶| Generator |────────▶| IMG |────────▶|  Discrim  |────────▶0.2
    | |         |           |         └─────┘         |           |
    └─┘         └───────────┘                         └───────────┘       
   Vector           Update                                Fixed

本程序数据集结构: 
    ├── /Pytorch-Knowledge/
    │  └── /dog/
    │    ├── /0/    // 存放生成器生成的图片
    │    ├── /1/    // 存放真实的柴犬图片
    │       ├── image_0.jpeg 
    │       ├── image_1.jpeg
    │       ...
    │       └──image_99.jpeg

本程序鉴别器网络结构:
┌────────────────────────────────────────────────────────────────────────────────────────────┐
|                                                                                            |
| ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐
| | IMG |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶Sigmoid
| └─────┘ (4x4,2) └─────┘         └─────┘         └─────┘         └─────┘
|  Input 
| (96x96) 
└────────────────────────────────────────────────────────────────────────────────────────────┘ 
"""

def imshow(inputs, picname):
    plt.ion
    inputs = inputs / 2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.pause(0.01)
    plt.savefig(picname+".jpg")
    plt.close()

class D(nn.Module):
    def __init__(self, nc, ndf) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(nn.Linear(256*6*6, 1), nn.Sigmoid())

data_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

trainset = datasets.ImageFolder('dog', data_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,shuffle=True, num_workers=0)

inputs, _ = next(iter(trainloader))
imshow(torchvision.utils.make_grid(inputs), "RealDataSample")
