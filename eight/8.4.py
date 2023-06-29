# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 8.4
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/6/5 10:20
@Version: 
@License: 
@Reference: 
@History:
- 2023/6/5 10:20:
==================================================  
"""
__author__ = 'zxx'

import torch
from d2l import torch as d2l


def main():
    X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
    H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
    print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))
    print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))


if __name__ == '__main__':
    main()
