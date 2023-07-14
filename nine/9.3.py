# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 9.3
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/14 9:54
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/14 9:54:
==================================================  
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l

"""
在深度循环神经网络中，隐藏状态的信息被传递到当前层的下一时间步和下一层的当前时间步；
有许多不同风格的深度循环神经网络，如长短期记忆网络、门控循环单元或经典循环神经网络。这些模型在深度学习框架的高级API中都有涵盖；
总体而言，深度循环神经网络需要大量的工作（如学习率和修剪）来确保合适的收敛，模型的初始化也需要谨慎。
"""


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, num_layer = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layer)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    num_epochs, lr = 500, 2
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    d2l.plt.show()


if __name__ == '__main__':
    main()
