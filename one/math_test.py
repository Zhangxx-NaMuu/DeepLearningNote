import torch
import numpy as np
from IPython import display
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt


# x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# # cumsum()函数按照某种规则计算和，不改变张量的shape
# y = x.cumsum(axis=0)
# z = torch.sum(x * y)


# print(x)
# # z = np.dot(x, y)
# print(y)
# print(z)
def f(x):
    return 3 * x ** 2 - 4 * x


# 求极限
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    plt.show()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果 `X` 有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# 非标量变量的反向传播
def non_scalar_back_propagation():
    x = torch.arange(4.0, requires_grad=True)
    y = x * x
    y.sum().backward()
    print(x.grad)


if __name__ == '__main__':
    # h = 0.1
    # for i in range(5):
    #     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    #     h *= 0.1
    # x = np.arange(0, 3, 0.1)
    # plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    non_scalar_back_propagation()
