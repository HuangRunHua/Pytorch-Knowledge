import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    # 存储训练/测试数据的路径，若“data”文件夹不存在则创建一个data文件夹
    root="data",    
    # 指定训练或测试数据集，True表示训练集
    train=True,
    # 如果数据在根目录下不可用，则从 Internet 下载数据
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

print(training_data)