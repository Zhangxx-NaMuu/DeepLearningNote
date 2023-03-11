# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DL_learnning -> dropout_layer
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2022/12/16 16:47
@Version: V0.1
@License: 
@Reference: dropout层
@History:
- 2022/12/16 16:47:
==================================================
"""
__author__ = 'zxx'

import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)


X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))

        # 只有在训练模型时才使用dropout
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()

print("#######################简洁实现############################")
net1 = nn.Sequential(nn.Flatten(),
                     nn.Linear(784, 256),
                     nn.ReLU(),
                     # 在第一个全连接层之后添加dropout层
                     nn.Dropout(0.4),
                     nn.Linear(256, 256),
                     nn.ReLU(),
                     # 在第二个全连接层之后添加dropout层
                     nn.Dropout(0.1),
                     nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net1.apply(init_weights)
trainer = torch.optim.SGD(net1.parameters(), lr=lr)
d2l.train_ch3(net1, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()
