# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 8.1
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/5/20 10:19
@Version: 
@License: 
@Reference: 
@History:
- 2023/5/20 10:19:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l

"""
使用正弦函数和一些可加性噪声来生成序列数据，时间步为1，2，……，1000
"""


def main():
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()
    tua = 4
    features = torch.zeros((T - tua, tua))
    for i in range(tua):
        features[:, i] = x[i:, T - tua + 1]
    labels = x[tua:].reshape((-1, 1))
    batch_size, n_train = 16, 600
    # 只有前n_train个样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    loss = nn.MSELoss()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)

if __name__ == '__main__':
    main()
