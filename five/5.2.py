# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 5.2
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/3/21 17:59
@Version: 
@License: 
@Reference: 
@History:
- 2023/3/21 17:59:
==================================================
"""
__author__ = 'zxx'

import torch
from torch import nn


# #########################5.2.1参数访问############################
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


def main1(net, X):
    # net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    # X = torch.rand(size=(2, 4))
    print(net(X))
    # 检查第二个全连接层的参数
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)
    # 以访问参数的梯度
    print(net[2].weight.grad is None)
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    print(net.state_dict()['2.bias'].data)

    # 嵌套
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet)
    print(rgnet(X))
    print(rgnet[0][1][0].bias.data)


# #########################5.2.2 参数初始化############################
def init_normal(m):
    """
    内置初始化
    :param m:
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    """
    将所有参数初始化为常熟
    :param m:
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


def main2(net2, x):
    net2.apply(init_normal)
    print(net2[0].weight.data[0])
    print(net2[0].bias.data[0])
    net2.apply(init_constant)
    print(net2[0].weight.data[0])
    print(net2[0].bias.data[0])
    net2[0].apply(xavier)
    net2[2].apply(init_42)
    print(net2[0].weight.data[0])
    print(net2[2].weight.data)
    net2.apply(my_init)
    print(net2[0].weight[:2])


# #########################5.2.3 参数绑定############################

def main3(x):
    # 定义一个共享层，以便引用他的参数
    shared = nn.Linear(8, 8)
    net2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         shared, nn.ReLU(),
                         shared, nn.ReLU(),
                         nn.Linear(8, 1))
    print(net2(x))
    # 检查参数是否相同
    print(net2[2].weight.data[0] == net2[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    # 确保他们实际上是同一个对象，而不只是有相同的值
    print(net2[2].weight.data[0] == net2[4].weight.data[0])


if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    # main2(net, X)
    main3(X)

