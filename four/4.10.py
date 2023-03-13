# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 4.10
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/3/11 11:18
@Version: 
@License: 
@Reference: 
@History:
- 2023/3/11 11:18:
==================================================
(一)本代码的质量保证期（简称“质保期”）为上线内 1个月，质保期内乙方对所代码实行包修改服务。
(二)本代码提供三包服务（包阅读、包编译、包运行）不包熟
(三)本代码所有解释权归权归佛祖所有，禁止未开光盲目上线
(四)请严格按照保养手册对代码进行保养，本代码特点：
     i. 运行在风电、水电的机器上
    ii. 机器机头朝东，比较喜欢太阳的照射
   iii. 集成此代码的人员，应拒绝黄赌毒，容易诱发本代码性能越来越弱
声明：未履行将视为自主放弃质保期，本人不承担对此产生的一切法律后果
如有问题，热线：114
                      _oo0oo_
                     o8888888o
                     88" . "88
                     (| -_- |)
                     0\  =  /0
                   ___/`---'\___
                 .' \\|     |// '.
                / \\|||  :  |||// \
               / _||||| -:- |||||- \
              |   | \\\  -  /// |   |
              | \_|  ''\---/''  |_/ |
              \  .-\__  '-'  ___/-. /
            ___'. .'  /--.--\  `. .'___
         ."" '<  `.___\_<|>_/___.' >' "".
        | | :  `- \`.;`\ _ /`;.`/ - ` : | |
        \  \ `_.   \_ __\ /__ _/   .-` /  /
    =====`-.____`.___ \_____/___.-`___.-'=====     
"""
__author__ = 'zxx'

import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# @save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    """
    下载一个DATA_HUB中的文件，返回本地文件名
    :param name:
    :param cache_dir:
    :return:
    """
    assert name in DATA_HUB, f"{name} 不存在于{DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
    """
    下载并解压zip/tar文件
    :param name:
    :param folder:
    :return:
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.split(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    """
    下载DATA_HUB中所有文件
    :return:
    """
    for name in DATA_HUB:
        download(name)


def data_preprocess(all_features):
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean() / x.std()))
    # 在标准化数据后，所有数据都意味着消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 用独热编码处理离散数据，即离散数据连续化; Dummy_na=True, 将‘na'缺失值视为有效的特征值，并为其创建指示符特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
    return all_features


def log_rmse(net, features, label):
    # 为了在取对数时进一步稳定该值，将小于1的值设为1
    clipped_pred = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_pred), torch.log(label)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


if __name__ == '__main__':
    DATA_HUB['kaggle_house_train'] = (  # @save
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (  # @save
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))
    print(train_data.shape)
    print(test_data.shape)
    # 查看前四个和后两个特征以及相应的标签
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    # 删除没用的特征ID
    all_feature = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    all_feature = data_preprocess(all_feature)
    # 加载训练数据和测试数据，训练标签
    n_train = train_data.shape[0]
    train_feature = torch.tensor(all_feature[:n_train].values, dtype=d2l.float32)
    test_feature = torch.tensor(all_feature[n_train:].values, dtype=d2l.float32)
    train_label = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
    # 损失函数
    loss = nn.MSELoss()
    in_feature = train_feature.shape[1]
    net = nn.Sequential(nn.Linear(in_feature, 1))
