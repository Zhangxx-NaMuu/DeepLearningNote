# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 9.1
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/13 14:12
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/13 14:12:
==================================================  
"""
__author__ = 'zxx'
"""
门控循环单元（GRU）：
门控循环神经网络可以更好的捕获时间步距离很长的序列上的依赖关系；
重置门有助于捕获序列中的短期依赖关系；
更新门有助于捕获序列中的长期依赖关系；
重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。
"""

import torch
from torch import nn
from d2l import torch as d2l


def get_params(vocab_size, num_hiddens, device):
    """
    初始化参数模型，从标准差为0.01的高斯分布中提取权重，偏置项设为0
    :param vocab_size:
    :param num_hiddens: 隐藏单元的数量
    :param device:
    :return: 实例化与更新门、重置门、候选隐藏状态和输出层相关的所有权重和偏置
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, devices = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, devices, get_params, init_gru_state, gru)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, devices)
    d2l.plt.show()
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(devices)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, devices)
    d2l.plt.show()


if __name__ == '__main__':
    main()
