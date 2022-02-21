import torch
import matplotlib.pyplot as plt

# @Function Produce_X(x)
# 扩充输入向量x的维度
# 向量x的长度为训练集的个数，x = [1,2,3]
# X = Produce_X(x) -> X = [1,1
#                          2,1
#                          3,1]
def Produce_X(x):
    x0 = torch.ones(x.numpy().size)
    X = torch.stack((x, x0), dim=1)
    return X

def train(epochs=1, learning_rate=0.01):
    for epoch in range(epochs):
        # output = inputs * w
        # 计算当前参数w预测的输出值
        output = inputs.mv(w)
        # 计算损失函数
        loss = (output - target).pow(2).sum()
        # 求解损失函数对w的梯度向量
        loss.backward()
        # 更新参数w
        w.data -= learning_rate*w.grad
        # w的梯度默认会叠加因此需要清空梯度值
        w.grad.zero_()
        if epoch % 80 == 0:
            draw(output,loss)
    return w, loss

def draw(output, loss):
    # 清空图像画布
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    # 打印loss值
    plt.text(0.5,0,'loss=%s' % (loss.item()), fontdict={'size':20, 'color':'red'})
    plt.pause(0.005)

# 训练样本，x为输入，y为输出
x = torch.Tensor([1.4, 5, 11, 16, 21])
y = torch.Tensor([14.4, 29.6, 62, 85.5, 113.4])

# plt.scatter(x.numpy(), y.numpy())
# plt.show()

X = Produce_X(x)

inputs = X
target = y
# 随机初始化参数向量w
# requires_grad=True表示w需要计算关于自己的梯度
w = torch.rand(2, requires_grad=True)

w, loss = train(10000, learning_rate=1e-4)
print("final loss:", loss.item())
print("weights:", w.data)
