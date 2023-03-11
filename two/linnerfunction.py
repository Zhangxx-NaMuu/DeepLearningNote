# -*- coding: UTF-8 -*-
"""
=================================================
@path   ：zxx_code -> linnerfunction
@IDE    ：PyCharm
@Author ：dell
@Date   ：2022/1/6 10:11
==================================================
"""
__author__ = 'dell'

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 生成数据集
true_w = torch.tensor([2, -3, 4], dtype=torch.float)
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据

def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))
