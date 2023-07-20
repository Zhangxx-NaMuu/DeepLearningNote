# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 10.1
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/17 11:20
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/17 11:20:
==================================================  
"""
__author__ = 'zxx'

import torch
from d2l import torch as d2l


# @save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """
    可视化注意力权重
    :param matrices: shape:(要显示的行数，要显示的列数，查询的数目，键的数目)
    :param xlabel:
    :param ylabel:
    :param title:
    :param figsize:
    :param cmap:
    :return:
    """
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()


if __name__ == '__main__':
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='keys', ylabel='Queries')
