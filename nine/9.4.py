# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 9.4
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/14 10:39
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/14 10:39:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l

"""
在双向循环神经网络中，每个时间步的隐藏状态由当前时间步的前后数据同时决定；
双向循环神经网络与概率图模型中的“前向-后向”算法有着惊人的相似性；
双向循环神经网络主要用于序列编码和给定双向上下文的观测估计；
由于梯度链更长，因此双向循环网络的训练成本非常高。
"""


def main():
    # 加载数据
    batch_size, num_steps, device = 32, 35, d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # 通过设置'bidirective=True'来定义双向LSTM模型
    vocab_size, num_hiddens, num_layers = len(vocab), 512, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    # 训练模型
    num_epochs, lr = 500, 1
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    main()
