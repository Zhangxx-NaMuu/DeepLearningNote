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


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch{epoch + 1},'
              f'loss:{d2l.evaluate_loss(net, train_iter, loss):f}')


def main():
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()
    tua = 4
    features = torch.zeros((T - tua, tua))
    for i in range(tua):
        features[:, i] = x[i: T - tua + i]
    labels = x[tua:].reshape((-1, 1))
    batch_size, n_train = 16, 600
    # 只有前n_train个样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)
    loss = nn.MSELoss()
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    # 单步预测
    onestep_preds = net(features)
    d2l.plot([time, time[tua:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
             legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()
    # 多步预测
    multistep_preds = torch.zeros(T)
    multistep_preds[:n_train + tua] = x[: n_train + tua]
    for i in range(n_train + tua, T):
        multistep_preds[i] = net(multistep_preds[i - tua:i].reshape((1, -1)))

    d2l.plot([time, time[tua:], time[n_train + tua:]], [x.detach().numpy(), onestep_preds.detach().numpy(),
                                                        multistep_preds[n_train + tua:].detach().numpy()], 'time', 'x',
             legend=['data', '1 - step preds', 'multistep_preds'], xlim=[1, 1000], figsize=(6, 3))

    d2l.plt.show()

    # 基于 k = 1, 4, 16, 64，通过对整个序列预测的计算，让我们更仔细地看⼀下k步预测的困难。
    max_steps = 64
    features = torch.zeros((T - tua - max_steps + 1, tua + max_steps))
    # 列 `i` (`i` < `tau`) 是来⾃ `x` 的观测
    # 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
    for i in range(tua):
        features[:, i] = x[i:i + T - tua - max_steps + 1]

    # 列 `i` (`i` >= `tau`) 是 (`i - tau + 1`)步的预测
    # 其时间步从 `i + 1` 到 `i + T - tau - max_steps + 1`
    for i in range(tua, tua + max_steps):
        features[:, i] = net(features[:, i - tua:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    d2l.plot([time[tua + i - 1: T - max_steps + i] for i in steps],
             [features[:, (tua + i - 1)].detach().numpy() for i in steps], 'time', 'x',
             legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
             figsize=(6, 3))
    d2l.plt.show()


if __name__ == '__main__':
    main()

"""
* 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于你所拥有的序列数据，
在训练时始终要尊重其时间顺序，永远不要基于未来的数据进行训练。
* 序列模型的估计需要专门的统计工具，两种较流形的选择是自回归模型和隐变量自回归模型。
* 对于时间是向前推进的因果模型， 正向估计通常比反向估计更容易。
*对于直到时间步t的观测序列，其在时间步t+k的预测输出是’k步预测‘。随着我们对预测时间k值的增加，会造成误差的快速积累和预测质量的急速下降。
"""