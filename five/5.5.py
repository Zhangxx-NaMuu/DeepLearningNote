# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 5.5
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/3 17:01
@Version: 
@License: 
@Reference: 
@History:
- 2023/4/3 17:01:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn


# ###########################5.5.1 计算设备#######################
def main1():
    print(torch.device('cpu'))
    print(torch.cuda.device('cuda'))
    print(torch.cuda.device_count())
    print(try_gpu())
    print(try_gpu(10))
    print(try_all_gpus())


# 现在我们定义了两个⽅便的函数，这两个函数允许我们在请求的GPU不存在的情况下运⾏代码
def try_gpu(i=0):  # @save
    # 如果存在，则返回gpu(i)，否则返回cpu
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():  # @save
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# ###########################5.5.2 张量与gpu#######################
# 默认情况下，张量是再CPU上创建的
def main2():
    x = torch.tensor([1, 3, 4])
    print(x.device)
    # 再GPU上创建
    X = torch.ones(2, 3, device=try_gpu())
    print(X)
    Z = x.cuda(0)
    print(X)
    print(Z)
    print(X + Z)
    # 假设变量x已经存在GPU上。如果我们还是调⽤X.cuda(0)怎么办？它将返回X，而不会复制并分配新内存。
    print(X.cuda(0) is X)


# ###########################5.5.3 神经⽹络与GPU#######################
def main3():
    X = torch.ones(2, 3, device=try_gpu())
    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device=try_gpu())
    print(net(X))
    print(net[0].weight.data.device)


if __name__ == '__main__':
    main1()
    main2()
    main3()
