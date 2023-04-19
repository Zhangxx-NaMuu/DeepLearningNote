# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 7.3
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/19 17:09
@Version: 
@License: 
@Reference: NiN
@History:
- 2023/4/19 17:09:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


def main():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
        nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是0
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        nn.AdaptiveMaxPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为（批量大小，10）
        nn.Flatten())

    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'Output shape:\t', X.shape)
    return net


if __name__ == '__main__':
    net = main()
    lr, num_epochs, batch_size = 0.1, 10, 64
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    d2l.plt.show()
