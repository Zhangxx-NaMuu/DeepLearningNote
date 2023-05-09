# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 7.5
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/23 16:28
@Version: 
@License: 
@Reference: 批量归一化
@History:
- 2023/4/23 16:28:
==================================================  
"""
__author__ = 'zxx'

"""
为什么需要批量归一化层：
1、数据预处理得方式会对最终的结果产生巨大的影响，归一化可以很好得配合优化器使用，可以将参数得量级进行统一；
2、便于网络收敛；
3、改善过拟合，对于深层得网络容易过拟合，可以用批量归一化来改善。

批量归一化可以应用到单一层也可用到所有层，原理如下：
在每次训练迭代中，我们首先归一化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。接下来我们应用比例系数和比例偏移。
正是由于这个基于批量统计得标准化，才有了批量归一化得名称。
注：如果使用大小为1得批量应用批量归一化，将无法学到任何东西。这是因为在减去均值之后，每个隐藏单元将为0.只有使用足够大的小批量，
批量归一化这种方法才是有效且稳定的。

全连接层：通常将批量归一化置于全连接层中的仿射变换和激活函数之间
卷积层：可以再卷积层之后和非线性激活函数之前应用批量归一化
"""

import torch
from torch import nn
from d2l import torch as d2l


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过`is_grad_enabled`来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维度上（axis=1）的均值和方差；这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 在训练模式下，使用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features：全连接层的输出数量或卷积层的输出通道数
    # num_dim：2表示全连接层，4表示卷积层
    def __init__(self, num_features, num_dim):
        super().__init__()
        if num_dim == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数， 分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在的显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return Y


def main():
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dim=4), nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dim=4), nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dim=2), nn.Sigmoid(),
        nn.Linear(120, 84), BatchNorm(84, num_dim=2), nn.Sigmoid(),
        nn.Linear(84, 10))
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()
    print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))


if __name__ == '__main__':
    main()
