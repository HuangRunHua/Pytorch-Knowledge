import os
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
| ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐   Cov   ┌─────┐                    |
| | IMG |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶Sigmoid    |
| └─────┘ (4x4,2) └─────┘ (4x4,2) └─────┘ (4x4,2) └─────┘ (4x4,2) └─────┘                    |
|  Input                                                           Output                    |
| (96x96)                                                          (6x6)                     | 
└────────────────────────────────────────────────────────────────────────────────────────────┘

本程序生成器网络结构:
┌───────────────────────────────────────────────────────────────────────────────────────────┐
| ┌─┐                                                                         ┌───────────┐ |         
| | |         ┌─────┐ Deconv  ┌─────┐ Deconv  ┌─────┐ Deconv  ┌─────┐ Deconv  |           | |
| | |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶| IMG |────────▶|    IMG    | |
| | |         └─────┘ (4x4,2) └─────┘ (4x4,2) └─────┘ (4x4,2) └─────┘ (4x4,2) |           | |
| └─┘                                                                         └───────────┘ |       
| Vect                                                                                      |
└───────────────────────────────────────────────────────────────────────────────────────────┘
"""

def imshow(inputs, picname):
    plt.ion
    inputs = inputs / 2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.pause(0.01)
    plt.savefig(picname+".jpg")
    plt.close()

"""
┌────────────────────────────────────┐
|       Define Discriminator         |
└────────────────────────────────────┘
"""
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

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 256*6*6)
        out = self.fc(out)
        return out

"""
┌────────────────────────────────────┐
|          Define Generator          |
└────────────────────────────────────┘
"""
class G(nn.Module):
    def __init__(self, nc, ngf, nz, feature_size) -> None:
        super().__init__()
        self.prj = nn.Linear(feature_size, nz*6*6)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2), nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf), nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        out = self.prj(x)
        out = out.view(-1, 1024, 6, 6)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=100, print_every=10):
    iter_count = 0
    for epoch in range(epochs):
        for inputs, _ in trainloader:
            real_inputs = inputs
            fake_inputs = g(torch.rand(5, 100))

            real_labels = torch.ones(real_inputs.size(0))
            fake_labels = torch.zeros(5)

            real_outputs = d(real_inputs)
            d_loss_real = criterion(real_outputs, real_labels)
            real_scores = real_outputs

            fake_outputs = d(fake_inputs)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            fake_scores = fake_outputs

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_inputs = g(torch.randn(5, 100))
            outputs = d(fake_inputs)
            real_labels = torch.ones(outputs.size(0))
            g_loss = criterion(outputs, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 设定每100次显示生成的图片
            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G: {:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
                pic_name = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
                imshow(torchvision.utils.make_grid(fake_inputs.date), pic_name)

            # 设定每10次打印一次损失值
            if (iter_count % print_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G: {:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
                pic_name = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
            iter_count += 1


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

d = D(3, 32)
g = G(3, 128, 1024, 100)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0003)

train(d, g, criterion, d_optimizer, g_optimizer, epochs=300)