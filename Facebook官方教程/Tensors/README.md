# 张量(Tensors)

张量是一种特殊的数据结构，与数组和矩阵非常相似。 在 PyTorch 中，我们使用张量来编码模型的输入和输出，以及模型的参数。

张量类似于 [NumPy](https://numpy.org/) 的 `ndarray`，除了张量可以在 GPU 或其他硬件加速器上运行。 事实上，张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据（参见 Bridge with NumPy）。 张量还针对自动微分进行了优化（我们将在稍后的 Autograd 部分中看到更多相关内容）。 如果您熟悉 ndarrays，那么您对 Tensor API 会很熟悉。 如果没有，请跟随！

```python
import torch
import numpy as np
```

## 初始化张量

张量可以以各种方式初始化。 请看以下示例：

**直接从数据**

张量可以直接从数据中创建。 数据类型是自动推断的。
```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

**来自 NumPy 数组**

张量可以从 NumPy 数组中创建（反之亦然 - 请参阅 Bridge with NumPy）。

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**从另一个张量：**

新张量保留参数张量的属性（形状、数据类型），除非显式覆盖。

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

输出:
```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.4557, 0.7406],
        [0.5935, 0.1859]])
```

**使用随机或恒定值：**

`shape` 是张量维度的元组。 在下面的函数中，它决定了输出张量的维度。
```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

输出:
```bash
Random Tensor:
 tensor([[0.8012, 0.4547, 0.4156],
        [0.6645, 0.1763, 0.3860]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## 张量的属性

张量属性描述了它们的形状、数据类型和存储它们的设备。
```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

输出：
```bash
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

## 张量运算
[这里](https://pytorch.org/docs/stable/torch.html)全面介绍了超过 100 种张量运算，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等。

这些操作中的每一个都可以在 GPU 上运行（通常以比 CPU 更高的速度）。 如果您使用的是 Colab，请转到运行时 > 更改运行时类型 > GPU 来分配 GPU。

默认情况下，张量是在 CPU 上创建的。 我们需要使用 .to 方法（在检查 GPU 可用性之后）将张量显式移动到 GPU。 请记住，跨设备复制大张量在时间和内存方面可能会很昂贵！
```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
```

尝试列表中的一些操作。 如果您熟悉 NumPy API，您会发现 Tensor API 使用起来轻而易举。

**标准的类似 numpy 的索引和切片：**
```python
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

输出:
```bash
First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**连接张量** 您可以使用 `torch.cat` 沿给定维度连接一系列张量。 另请参阅 `torch.stack`，另一个与 `torch.cat` 略有不同的张量加入 op。

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

输出:
```bash
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**算术运算**
```python
# 这计算了两个张量之间的矩阵乘法。 y1, y2, y3 将具有相同的值
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# 这将计算元素乘积。 z1、z2、z3 将具有相同的值
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

**单元素张量** 如果您有一个单元素张量，例如通过将张量的所有值聚合为一个值，您可以使用 `item()` 将其转换为 `Python` 数值：
```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

输出:
```python
12.0 <class 'float'>
```

**就地操作** 将结果存储到操作数中的操作称为就地操作。 它们用 `_` 后缀表示。 例如：`x.copy_(y)`，`x.t_()`，会改变`x`。
```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

输出:
```python
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

> ⚠️ 就地操作可以节省一些内存，但在计算导数时可能会出现问题，因为会立即丢失历史记录。 因此，不鼓励使用它们。

## Bridge with NumPy
CPU 和 NumPy 数组上的张量可以共享它们的底层内存位置，改变一个会改变另一个。

**张量到 NumPy 数组**
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

输出:
```python
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

张量的变化反映在 NumPy 数组中。
```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

输出:
```python
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

**NumPy 数组到张量**
```python
n = np.ones(5)
t = torch.from_numpy(n)
```
NumPy 数组的变化反映在张量中。

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

输出:
```python
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```
