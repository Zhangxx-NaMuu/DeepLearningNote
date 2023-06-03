# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：DeepLearningNote -> 8.3
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/5/20 16:37
@Version: 
@License: 
@Reference: 
@History:
- 2023/5/20 16:37:
==================================================  
"""
__author__ = 'zxx'

import random
import torch
from d2l import torch as d2l


def main():
    tokens = d2l.tokenize(d2l.read_time_machine())
    # 因为每个⽂本⾏不⼀定是⼀个句⼦或⼀个段落，因此我们把所有⽂本⾏拼接到⼀起
    corpus = [token for line in tokens for token in line]
    vocab = d2l.Vocab(corpus)
    print(vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
    d2l.plt.show()

    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
             ylabel='frequency: n(x)', xscale='log', yscale='log',
             legend=['unigram', 'bigram', 'trigram'])
    d2l.plt.show()
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)


def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行区分，随即范围包括‘num_steps - 1'
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度未’num_steps'的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来⾃两个相邻的、随机的、⼩批量中的⼦序列不⼀定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从`pos`位置开始的⻓度为`num_steps`的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这⾥，`initial_indices`包含⼦序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):  # @save
    """使⽤顺序分区⽣成⼀个⼩批量⼦序列。"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  # @save
    """加载序列数据的迭代器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


if __name__ == '__main__':
    main()
