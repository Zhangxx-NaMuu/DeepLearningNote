# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 6.5
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/11 9:51
@Version: 
@License: 
@Reference: 池化层：1、降低卷积层对位置的敏感性；2、降低对空间采样表示的敏感性
@History:
- 2023/4/11 9:51:
==================================================  
"""
__author__ = 'zxx'

"""
与卷积层的输入和卷积核之间的互相关运算不同的是：汇聚层不包含参数
"""

import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


def main():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))


print("###############################6.5.2填充和步幅####################################")
"""
与卷积层一样，池化层（汇聚层）可以通过填充和步幅改变输出形状
"""


def main1():
    # 构造一个有四个维度的输入张量X， 其中样本数和通道数都是1
    # TODO:原书中的dtype=d2l.float32报错
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)
    # 默认情况下，深度学习框架中的步幅与池化窗口的大小相同， 因此如果我们使用形状为（3，3）的池化窗口，那么得到的步幅形状为（3，3）
    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))
    # 填充和步幅可以手动设置
    pool2d1 = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d1(X))
    # 也可以设定以设定一个任意大小的矩形池化窗口，并分别设置填充和步幅的高度和宽度
    pool2d2 = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
    print(pool2d2(X))
    # 连结张量X 和 X + 1，构建具有两个通道的输入
    X = torch.cat((X, X + 1), 1)
    print(X)
    print(pool2d1(X))


if __name__ == '__main__':
    main()
    main1()
