# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 10.2
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/17 14:00
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/17 14:00:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # `queries` 和 `attention_weights` 的形状为 (查询个数, “键－值”对个数
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        # `values` 的形状为 (查询个数, “键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


if __name__ == '__main__':
    n_train = 50  # 训练样本数
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 训练样本的输入
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    print(len(x_test))  # 测试样本数
    n_test = len(x_test)
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)
    # `X_repeat` 的形状: (`n_test`, `n_train`),
    # 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # `x_train` 包含着键。`attention_weights` 的形状：(`n_test`, `n_train`),
    # 每⼀⾏都包含着要在给定的每个查询的值（`y_train`）之间分配的注意⼒权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # `y_hat` 的每个元素都是值的加权平均值，其中的权重是注意⼒权重
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)
    d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')
    d2l.plt.show()

    # 的批量矩阵乘法
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))
    print(torch.bmm(X, Y).shape)

    weights = torch.ones((2, 10)) * 0.1
    values = torch.arange(20.0).reshape((2, 10))
    print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

    # 训练模型
    # `X_tile` 的形状: (`n_train`, `n_train`), 每⼀⾏都包含着相同的训练输⼊
    X_tile = x_train.repeat((n_train, 1))
    # `Y_tile` 的形状: (`n_train`, `n_train`), 每⼀⾏都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # `keys` 的形状: ('n_train', 'n_train' - 1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # `values` 的形状: ('n_train', 'n_train' - 1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad()
        # 注意：L2 Loss = 1/2 * MSE Loss。
        # PyTorch 的 MSE Loss 与 MXNet 的 L2Loss 差⼀个 2 的因⼦，因此被除2。
        l = loss(net(x_train, keys, values), y_train) / 2
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))
    d2l.plt.show()
    # `keys` 的形状: (`n_test`, `n_train`), 每⼀⾏包含着相同的训练输⼊（例如：相同的键）
    keys = x_train.repeat((n_test, 1))
    # `value` 的形状: (`n_test`, `n_train`)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    plot_kernel_reg(y_hat)

    d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')
    d2l.plt.show()

