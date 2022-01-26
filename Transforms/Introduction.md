# 数据转换
数据并不总是以训练机器学习算法所需的最终处理形式出现。 我们使用转换来对数据进行一些操作并使其适合训练。

所有 TorchVision 数据集都有两个参数 - 用于修改特征的`transform`和用于修改标签的 `target_transform` - 接受包含转换逻辑的可调用对象。[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) 模块提供了几个开箱即用的常用转换。

FashionMNIST 特征是 PIL 图像格式，标签是整数。 对于训练，我们需要将特征作为归一化张量，并将标签作为 one-hot 编码张量。one-hot 编码过程参考[示例代码](https://github.com/HuangRunHua/Pytorch-Knowledge/blob/main/Transforms/one_hot_example.py)。这里简要解释one-hot编码要达到的目的。

假设6个数据集如下，其中`[0,3,2,3,1,1]`分别表示数据集所属于的类别，同时假设数据集的种类为`num_classes = 4`种。经过编码后最终得到的数据集矩阵为：
```python
[ 
    [1. 0. 0. 0.]   标签为0的数据集
    [0. 0. 0. 1.]   标签为3的数据集
    [0. 0. 1. 0.]   标签为2的数据集
    [0. 0. 0. 1.]   标签为3的数据集
    [0. 1. 0. 0.]   标签为1的数据集
    [0. 1. 0. 0.]   标签为1的数据集
]  
```
为了进行这些转换，我们使用 `ToTensor` 和 `Lambda`。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # value=1将会使得被选中的索引处的值为1
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

输出:
```python
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## ToTensor()
[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor) 将 `PIL` 图像或 `NumPy ndarray` 转换为 `FloatTensor`。 并在 [0., 1.] 范围内缩放图像的像素强度值。

简单来说，`ToTensor()`将`shape`为`(H, W, C)`的`nump.ndarray`或`img`转为`shape`为`(C, H, W)`的`tensor`，其将每一个数值归一化到`[0,1]`。过程如下：
- 先由HWC转置为CHW格式；
- 再转为float类型；
- 最后，每个像素除以255。

## Lambda 变换

Lambda 转换应用任何用户定义的 lambda 函数。 在这里，我们定义了一个函数来将整数转换为 one-hot 编码张量。 它首先创建一个大小为 10（我们数据集中的标签数量）的零张量，并调用 [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)，它在标签 y 给定的索引上分配 `value=1`。

```python
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

`.scatter(dim, index, src)`的用法可以参考[示例代码](https://github.com/HuangRunHua/Pytorch-Knowledge/blob/main/Transforms/scatter_example.py)，详细介绍可以参考[pytorch 深入理解 tensor.scatter_ ()用法](https://blog.csdn.net/ao1886/article/details/107749007)或[官方文档](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)。

## 延伸阅读
- [torchvision.transforms API](https://pytorch.org/vision/stable/transforms.html)
