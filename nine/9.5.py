# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 9.5
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/14 11:07
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/14 11:07:
==================================================  
"""
__author__ = 'zxx'

import os
import torch
from d2l import torch as d2l

"""
机器翻译指的是将文本序列从一种语言自动翻译成另一种语言；
使用单词级词元化的词汇量，将明显大于使用字符级词元化时的词汇量，为了缓解这一问题，我们可以将低频词元视为相同的未知词元；
通过截断和填充文本序列，可以保证所有的文本序列都具有相同的长度，以便以小批量的方式加载；
"""


# @save
def read_data_nmt():
    # 载入“英语-法语”数据集
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()


# @save
def preprocess_nmt(text):
    # 预处理“英语-法语”数据集
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


# @save
def tokenize_nmt(text, num_examples=None):
    # 词元化“英语-法语‘数据集
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# @save
def truncate_pad(line, num_steps, padding_token):
    # 截断或填充文本序列
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))  # 填充


# @save
def build_array_nmt(lines, vocab, num_steps):
    # 将机器翻译的文本序列转换成小批量
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# @save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    # 返回翻译数据集的迭代器和词汇表
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos]'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos]'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def main():
    # 下载⼀个由Tatoeba项⽬的双语句⼦对114组成的“英－法”数据集，数据集中的每⼀⾏都是制表符分隔
    # ⽂本序列对，序列对由英⽂⽂本序列和翻译后的法语⽂本序列组成
    d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
    raw_text = read_data_nmt()
    print(raw_text[:75])
    text = preprocess_nmt(raw_text)
    print(text[:80])
    source, target = tokenize_nmt(text)
    print(source[:6], target[:6])
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]], label=['source', 'target'])
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(loc='upper right')  # plt.legend()将label设置显示在右上角
    d2l.plt.show()

    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos]'])
    print(len(src_vocab))
    print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('valid lengths for X:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('valid lengths for Y:', Y_valid_len)
        break


if __name__ == '__main__':
    main()
