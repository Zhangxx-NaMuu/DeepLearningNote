# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 8.2
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/5/20 15:34
@Version: 
@License: 
@Reference: 文本预处理
@History:
- 2023/5/20 15:34:
==================================================  
"""
__author__ = 'zxx'

import collections
import re
from d2l import torch as d2l

"""
下⾯的 tokenize 函数将⽂本⾏列表作为输⼊，列表中的每个元素是⼀个⽂本序列（如⼀条⽂本⾏）。每个
⽂本序列⼜被拆分成⼀个词元列表，词元（token）是⽂本的基本单位。最后，返回⼀个由词元列表组成的列
表，其中的每个词元都是⼀个字符串（string）。
"""


def tokenize(lines, token='word'):  # @save
    # 将文本拆分为单词或字符词元
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def read_time_machine():  # @save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z] +', ' ', line).strip().lower() for line in lines]


class Vocab:  # @save
    # 文本词汇表
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):

        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)

        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(tokens):  # @save
    # 统计次元频率
    # 这里的’tokens‘是1D或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成使用词元填充的一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):  # @save
    """返回时光机器数据集的词元索引列表和词汇表。"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，
    # 所以将所有⽂本⾏展平到⼀个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def main():
    # @save
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
    lines = read_time_machine()
    print(f'# text lines: {len(lines)}')
    print(lines[0])
    print(lines[10])

    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])
    for i in [0, 10]:
        print('words:', tokens[i])
        print('indices:', vocab[tokens[i]])

    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))


if __name__ == '__main__':
    main()
