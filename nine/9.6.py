# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 9.6
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/14 16:46
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/14 16:46:
==================================================  
"""
__author__ = 'zxx'

from torch import nn


# @save
class Encoder(nn.Module):
    # 编码器-解码器的基本编码器接口
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


# @save
class Decoder(nn.Module):
    # 编码器-解码器结构的基本解码器接口
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


# 合并编码器-解码器
# @save
class EncoderDecoder(nn.Module):
    # 编码器-解码器结构的基类
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, dec_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
