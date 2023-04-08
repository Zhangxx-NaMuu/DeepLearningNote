# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 6.3
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/8 14:35
@Version: 
@License: 
@Reference: 
@History:
- 2023/4/8 14:35:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn


def comp_conv2d(conv2d, X):
    """
    为了方便起见，我们定义了一个计算卷积层的函数。
    此函数初始化卷积层权重，并对输入和输出提高和所见相应的维数
    :param conv2d:
    :param X:
    :return:
    """
    # 这里（1， 1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


def manin():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    print(comp_conv2d(conv2d, X).shape)
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    print(comp_conv2d(conv2d, X).shape)


if __name__ == '__main__':
    manin()
