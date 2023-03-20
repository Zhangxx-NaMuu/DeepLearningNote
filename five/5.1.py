# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 5.1
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/3/18 9:28
@Version: 
@License: 
@Reference: 
@History:
- 2023/3/18 9:28:
==================================================
"""
__author__ = 'zxx'

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            """
            这里block是Module子类的一个实例，我们把他保存在Module类的成员变量_modules中，block的类型是OrderedDict
            """
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weights = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和dot函数
        X = F.relu(torch.mm(X, self.rand_weights) + 1)
        # 复用全连接层，这相当于两个全连接层共享参数
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


if __name__ == '__main__':
    X = torch.rand(2, 20)
    net = MLP()
    net1 = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    net2 = FixedHiddenMLP()
    net3 = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(net3(X))
    print(net2(X))

    print(net1(X))
    print(net(X))
