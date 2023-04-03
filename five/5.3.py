# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 5.3
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/1 14:25
@Version: 
@License:
@Reference: 小结：1、可以通过基本层类设计自定义层; 2、在自定义层完成后，可以在任意环境和网络结构中调用该自定义层；3、层可以有局部参数，这些参数可以通过内置函数创建
@History:
- 2023/4/1 14:25:
==================================================  
"""
__author__ = 'zxx'

import torch
import torch.nn.functional as F
from torch import nn


# ################5.3.1 不带参数的层###################
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


def main1():
    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
    # 将层作为组件合并到模型中
    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())


# ################5.3.2 带参数的层###################
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


def main2():
    # 实例化MyLinear类并访问其参数
    linear = MyLinear(5, 3)
    print(linear.weight)
    # 使用自定义层直接执行正向传播计算
    print(linear(torch.rand(2, 5)))
    # 使用自定义层构建模型，像使用内置的全连接层一样使用自定义层
    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))


if __name__ == '__main__':
    main1()
    main2()
