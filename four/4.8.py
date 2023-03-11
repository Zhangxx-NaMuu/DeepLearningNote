# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DL_learnning -> 4.8
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/1/6 17:02
@Version: 
@License: 
@Reference: 梯度消失
@History:
- 2023/1/6 17:02:
==================================================
"""
__author__ = 'zxx'

import torch
from d2l import torch as d2l

print('######################梯度消失#########################')
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.show()

print('#####################梯度爆炸#########################')
M = torch.normal(0, 1, size=(4, 4))
print("一个矩阵\n", M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print("乘以100个矩阵后\n", M)

