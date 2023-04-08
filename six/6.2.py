# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 6.2
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/6 18:56
@Version: 
@License: 
@Reference: 
@History:
- 2023/4/6 18:56:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, k):  # @save
    """
    计算二维互相关运算
    :param X:
    :param k:
    :return:
    """
    h, w = k.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * k).sum()
    return Y


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def main1():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    k = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, k))
    # 调用Conv2D
    conv = Conv2D((1, 2))
    print(conv(X))


def main2():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    # 我们构造⼀个⾼度为 1 、宽度为 2 的卷积核 K 。当进⾏互相关运算时，如果⽔平相邻的两元素相同，
    # 则输出为零，否则输出为⾮零
    k = torch.tensor([[1.0, -1.0]])
    # 输出Y中的1代表从⽩⾊到⿊⾊的边缘，-1代表从⿊⾊到⽩⾊的边缘，其他情况的输出为 0。
    Y = corr2d(X, k)
    print(Y)
    # 现在我们将输⼊的⼆维图像转置，再进⾏如上的互相关运算。其输出如下，之前检测到的垂直边缘消失了。
    # 不出所料，这个卷积核K只可以检测垂直边缘，⽆法检测⽔平边缘
    Y1 = corr2d(X.t(), k)
    print(Y1)


def main3():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    k = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, k)
    # 构造一个二维卷积层， 它具有1个输出通道和形状为（1， 2） 的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    # 这个二维卷积层使用四维输入和输出格式（批量大小，通道，高度，宽度）
    # 其中批量大小和通道数都为1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()

        # 迭代卷积核
        conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {l.sum():.3f}')
    # 查看所学的卷积核的权重张量
    print(conv2d.weight.data.reshape((1, 2)))


if __name__ == '__main__':
    # main1()
    # main2()

    main3()

