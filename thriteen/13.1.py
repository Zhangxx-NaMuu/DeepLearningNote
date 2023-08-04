# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 13.1
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/3 17:33
@Version: 
@License: 
@Reference: 
@History:
- 2023/8/3 17:33:
==================================================  
"""
__author__ = 'zxx'

import torch
import torchvision
from torch import nn
from d2l import torch as d2l


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()


def main():
    # all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
    # d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
    d2l.set_figsize()
    img = d2l.Image.open(
        r"C:\Users\dell\Documents\WeChat Files\wxid_w11gvvoh3yg922\FileStorage\File\2023-06\张新霞2\张新霞 (71).jpg")
    d2l.plt.imshow(img)
    d2l.plt.show()
    apply(img, torchvision.transforms.RandomHorizontalFlip())
    apply(img, torchvision.transforms.RandomVerticalFlip())


if __name__ == '__main__':
    main()
