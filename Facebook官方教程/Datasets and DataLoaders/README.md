# [数据集和数据加载器](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset)

处理数据样本的代码可能会变得混乱且难以维护； 理想情况下，我们希望我们的数据集代码与我们的模型训练代码分离，以获得更好的可读性和模块化。 PyTorch 提供了两个数据原语：`torch.utils.data.DataLoader` 和 `torch.utils.data.Dataset`，允许您使用预加载的数据集以及您自己的数据。 `Dataset` 存储样本及其对应的标签，`DataLoader` 在 `Dataset` 周围包装了一个可迭代对象，以便轻松访问样本。

PyTorch 域库提供了许多预加载的数据集（例如 FashionMNIST），这些数据集是 `torch.utils.data.Dataset` 的子类，并实现了特定于特定数据的功能。 它们可用于对您的模型进行原型设计和基准测试。 您可以在此处找到它们：[图像数据集](https://pytorch.org/vision/stable/datasets.html)、[文本数据集](https://pytorch.org/text/stable/datasets.html)和[音频数据集](https://pytorch.org/audio/stable/datasets.html)。


## 加载数据集

下面是如何从 TorchVision 加载 `Fashion-MNIST` 数据集的示例。 Fashion-MNIST 是 Zalando 文章图像的数据集，由 60,000 个训练示例和 10,000 个测试示例组成。 每个示例都包含 28×28 灰度图像和来自 10 个类别之一的相关标签。

我们使用以下参数加载 `FashionMNIST` 数据集：
- `root` 是存储训练/测试数据的路径，
- `train` 指定训练或测试数据集，
- `download=True` 如果数据在根目录下不可用，则从 Internet 下载数据。
- `transform` 和 `target_transform` 指定特征和标签转换

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

输出：
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
> 注意执行上述代码将会下载四个包：训练集与测试集的图片以及图片对应的标签

## 迭代和可视化数据集

我们可以像列表一样手动索引数据集：```training_data[index]```。 我们使用 matplotlib 来可视化我们训练数据中的一些样本。
```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    # randint返回一个随机的张量，张量的shape由参数size指定
    # 张量里的元素大小范围为0～len(training_data)
    # 使用 item() 将torch.randint(len(training_data), size=(1,))转换为 Python 可读的类型
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # imshow表示指定数据的展示方式为图片
    # cmap表示图片色调为灰色
    # .squeeze()表示对数据的维度进行压缩，将所有为1的维度删掉，不为1的维度没有影响
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_001.png)

## 为您的文件创建自定义数据集

自定义数据集类必须实现三个函数：`__init__`、`__len__` 和 `__getitem__`。 看看这个实现； FashionMNIST 图像存储在目录 `img_dir` 中，它们的标签分别存储在 CSV 文件 `annotations_file` 中。

在接下来的部分中，我们将分解每个函数中发生的事情。
```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # self.img_labels.iloc[idx, 0]表示选取第0列第idx个元素
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### __init__
`__init__` 函数在实例化 Dataset 对象时运行一次。 我们初始化包含图像、注释文件和两种转换的目录（在下一节中更详细地介绍）。

labels.csv 文件如下所示：
```python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```
经过`self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])`处理后，`self.img_labels`的输出为
```python
          file_name  label
0       tshirt1.jpg      0
1       tshirt2.jpg      0
......
1000  ankleboot999.jpg      9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    # read_csv实现了对annotations_file的读取，并给annotations_file按照列来给定names的标签
    self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### __len__
`__len__` 函数返回我们数据集中的样本数。
```python
def __len__(self):
    return len(self.img_labels)
```

### __getitem__
`__getitem__` 函数从给定索引 `idx` 的数据集中加载并返回一个样本。 根据索引，它识别图像在磁盘上的位置，使用 `read_image` 将其转换为张量，从 `self.img_labels` 中的 csv 数据中检索相应的标签，调用它们的转换函数（如果适用），并返回张量图像 和元组中的相应标签。

```python
def __getitem__(self, idx):
    # 假设self.img_dir = "Data", self.img_labels.iloc[1, 0] = "tshirt2.jpg"
    # 使用.join()函数后img_path = "Data/tshirt2.jpg"
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

## 使用 DataLoaders 为训练准备数据
数据集检索我们数据集的特征并一次标记一个样本。 在训练模型时，我们通常希望以“小批量”的形式传递样本，在每个 epoch 重新洗牌以减少模型过拟合，并使用 Python 的多处理来加速数据检索。

`DataLoader` 是一个迭代器，它通过一个简单的 API 为我们抽象了这种复杂性。

```python
from torch.utils.data import DataLoader

# shuffle=True表示打乱数据集
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## 遍历 DataLoader
我们已经将该数据集加载到 `DataLoader` 中，并且可以根据需要遍历数据集。 下面的每次迭代都会返回一批 `train_features` 和 `train_labels`（分别包含 `batch_size=64` 个特征和标签）。 因为我们指定了 `shuffle=True`，所以在我们遍历所有批次之后，数据会被打乱（为了更细粒度地控制数据加载顺序，请查看 [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)）。

```python
# Display image and label.
# iter()可将train_dataloader变为可迭代的迭代器
# next()表示开始遍历迭代器，初始情况下next()默认为第一个数据，如
# >>> l=[2,3,4]
# >>> iterl=iter(l)
# >>> iterl.next()
# 2
# >>> iterl.next()
# 3
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

![](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_002.png)

输出：
```python
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 6
```

## 延伸阅读
- [torch.utils.data API](https://pytorch.org/docs/stable/data.html)
