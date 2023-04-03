# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 5.4
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/4/3 15:57
@Version: 
@License: 
@Reference: 
@History:
- 2023/4/3 15:57:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from torch.nn import functional as F


# #####################5.4.1 加载和保存张量###########################
def main1():
    # 对于单个张量，直接调用save和load函数读写
    x = torch.arange(4)
    torch.save(x, 'x-file')

    x1 = torch.load('x-file')
    print(x1)

    # 存储张量并读回内存
    y = torch.zeros(4)
    torch.save([x, y], 'x-files')
    x2, y2 = torch.load('x-files')
    print(x2, y2)

    # 写入或读取将字符串映射到张量的字典
    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)


# #####################5.4.2 加载和保存模型参数###########################
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


def main2():
    net = MLP()
    X = torch.rand(size=(2, 20))
    Y = net(X)
    # 保存模型参数为‘mlp.params'
    torch.save(net.state_dict(), 'mlp.params')
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    print(clone.eval())
    Y_clone = clone(X)
    print(Y_clone == Y)


if __name__ == '__main__':
    main1()
    main2()
