import torch
import matplotlib.pyplot as plt
from time import perf_counter, time

def Produce_X(x):
    x0 = torch.ones(x.numpy().size)
    X = torch.stack((x, x0), dim=1)
    return X

def draw(output, loss):
    # 清空图像画布
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    # 打印loss值
    plt.text(0.5,0,'loss=%s' % (loss.item()), fontdict={'size':20, 'color':'red'})
    plt.pause(0.005)

def train(epochs=1, learning_rate=0.01):
    for epoch in range(epochs):
        # output = inputs * w
        # 计算当前参数w预测的输出值
        output = inputs.mv(w)
        # 加快收敛速度点小技巧
        loss = (output - target).pow(2).sum()/100000
        # 求解损失函数对w的梯度向量
        loss.backward()
        # 更新参数w
        w.data -= learning_rate*w.grad
        # w的梯度默认会叠加因此需要清空梯度值
        w.grad.zero_()
        if epoch % 80 == 0:
            draw(output,loss)
    return w, loss

# 产生100000个(-3,3)之间的点作为输入
x = torch.linspace(-3,3,100000)
X = Produce_X(x)

# 给真实的数据添加误差
y = x + 1.2*torch.rand(x.size())
w = torch.rand(2, requires_grad=True)

plt.scatter(x.numpy(), y.numpy(), s=0.001)
plt.show()

inputs = X
target = y

start  = perf_counter()
w, loss = train(10000, learning_rate=1e-4)
finish = perf_counter()
train_time = finish - start

print("Total Train Time: %s" % time)
print("final loss:", loss.item())
print("weights:", w.data)


