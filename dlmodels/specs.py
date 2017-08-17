# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

import inspect
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.legacy import nn as legnn

import layers


class LocalImport(object):

  def __init__(self, names):
    if not isinstance(names, dict):
      names = vars(names)
    self.names = names

  def __enter__(self):
    self.frame = inspect.currentframe()
    bindings = self.frame.f_back.f_globals
    self.old_bindings = {k: bindings.get(k, None) for k in self.names.keys()}
    bindings.update(self.names)

  def __exit__(self, some_type, value, traceback):
    del some_type, value, traceback
    bindings = self.frame.f_back.f_globals
    bindings.update(self.old_bindings)
    extras = [k for k, v in self.old_bindings.items() if v is None]
    for k in extras:
        del bindings[k]
    del self.frame


permitted_orders = [None] + "BLD BDL BDWH BWHD BD".split()


class Seq(nn.Sequential):
    def __init__(self, shape, order=None):
        nn.Sequential.__init__(self)
        assert order in permitted_orders
        self.value = Variable(torch.rand(*shape), volatile=True)
        self.order = order
        self.initial = self.value
        self.initial_order = self.order

    def dims(self):
        return self.order

    def check(self, order):
        assert order in permitted_orders
        if order is not None and self.order is not None:
            assert order == self.order

    def rank(self):
        return len(self.value.size())

    def size(self, index=None):
        if index is None:
            return self.value.size()
        else:
            return self.value.size(index)

    def add(self, modules, name=None, order=True):
        old_order = self.order
        if name is None:
            name = str(len(self._modules))
        if not isinstance(modules, list):
            modules = [modules]
        for m in modules:
            self.value = m(self.value)
            self.add_module(name, m)
        if order == True:
            self.order = old_order
        else:
            assert order in permitted_orders
            self.order = order
        return self

    def __repr__(self):
        result = "Input: %s %s\n" % (
            tuple(self.initial.size()), self.initial_order)
        result += nn.Sequential.__repr__(self) + "\n"
        result += "Output: %s %s\n" % (tuple(self.value.size()), self.order)
        return result


class Composable(object):
    def __init__(self, f, g=None):
        self.f = f
        self.g = g
        self.stacks = {}

    def __or__(self, g):
        return Composable(self, g)

    def __pow__(self, n):
        assert n > 0
        result = self
        for i in range(1, n):
            result = Composable(result, self)
        return result

    def __call__(self, *args, **kw):
        if self.g is None:
            return self.f(*args, **kw)
        else:
            return self.g(self.f(*args, **kw))

    def create(self, *args, **kw):
        base = Seq(args, order=kw.get("order"))
        return self.__call__(base)


def wrapup(f, order=True):
    def create(*args, **kw):
        def construct(arg):
            return arg.add(f(*args, **kw))
        return construct
    return create


Sig = wrapup(nn.Sigmoid)
Tanh = wrapup(nn.Tanh)
Relu = wrapup(nn.ReLU)
Smax = wrapup(nn.Softmax)
LSmax = wrapup(nn.LogSoftmax)
Reshape = wrapup(layers.Viewer)
PixelShuffle = wrapup(nn.PixelShuffle)
# Squeeze
# Expand
# Select


def Bn(*args, **kw):
    def construct(arg):
        rank = len(arg.size())
        if rank == 5:
            return arg.add(nn.BatchNorm3d(arg.size(1), *args, **kw))
        elif rank == 4:
            return arg.add(nn.BatchNorm2d(arg.size(1), *args, **kw))
        elif rank == 3:
            return arg.add(nn.BatchNorm1d(arg.size(1), *args, **kw))
        elif rank == 2:
            return arg.add(nn.BatchNorm1d(arg.size(1), *args, **kw))
        raise Exception("%d: bad rank for Bn")
    return Composable(construct)


def Mp(*args, **kw):
    def construct(arg):
        rank = len(arg.size())
        if rank == 3:
            return arg.add(nn.MaxPool3d(*args, **kw))
        elif rank == 4:
            return arg.add(nn.MaxPool2d(*args, **kw))
        elif rank == 3:
            return arg.add(nn.MaxPool1d(*args, **kw))
        raise Exception("%d: bad rank for Cl")
    return Composable(construct)


def Ap(*args, **kw):
    def construct(arg):
        rank = len(arg.size())
        if rank == 3:
            return arg.add(nn.AvgPool3d(*args, **kw))
        elif rank == 4:
            return arg.add(nn.AvgPool2d(*args, **kw))
        elif rank == 3:
            return arg.add(nn.AvgPool1d(*args, **kw))
        raise Exception("%d: bad rank for Cl")
    return Composable(construct)


def Cl(out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True):
    def construct(arg):
        ksize = kernel_size
        pad = padding
        assert arg.rank() > 2
        if isinstance(ksize, int):
            ksize = tuple([ksize] * (arg.rank() - 2))
        if pad is None:
            pad = tuple([x // 2 for x in ksize])
        rank = arg.rank()
        if rank == 5:
            return arg.add(nn.Conv3d(arg.size(1), out_channels, ksize,
                                     stride, pad, dilation, groups, bias))
        elif rank == 4:
            return arg.add(nn.Conv2d(arg.size(1), out_channels, ksize,
                                     stride, pad, dilation, groups, bias))
        elif rank == 3:
            return arg.add(nn.Conv1d(arg.size(1), out_channels, ksize,
                                     stride, pad, dilation, groups, bias))
        raise Exception("%d: bad rank for Cl")
    return Composable(construct)


def Cr(*args, **kw): return Cl(*args, **kw) | Relu()


def Ct(*args, **kw): return Cl(*args, **kw) | Tanh()


def Cs(*args, **kw): return Cl(*args, **kw) | Sig()


def Cbl(*args, **kw): return Cl(*args, bias=False, **kw) | Bn()


def Cbr(*args, **kw): return Cl(*args, bias=False, **kw) | Bn() | Relu()


def Cbt(*args, **kw): return Cl(*args, bias=False, **kw) | Bn() | Tanh()


def Cbs(*args, **kw): return Cl(*args, bias=False, **kw) | Bn() | Sig()


def Dws(n, **kw): return Cl(n, 1, **kw)


def Flat():
    def construct(arg):
        rank = len(arg.size())
        assert rank > 2
        new_depth = np.prod(list(arg.size())[1:])
        return arg.add(layers.Viewer(-1, new_depth))
    return Composable(construct)


def Fl(*args, **kw):
    def construct(arg):
        assert len(arg.size()) == 2
        return arg.add(nn.Linear(arg.size(1), *args, **kw))
    return Composable(construct)


def Fr(*args, **kw): return Fl(*args, **kw) | Relu()


def Ft(*args, **kw): return Fl(*args, **kw) | Tanh()


def Fs(*args, **kw): return Fl(*args, **kw) | Sig()


def Fbl(*args, **kw): return Fl(*args, bias=False, **kw) | Bn()


def Fbr(*args, **kw): return Fl(*args, bias=False, **kw) | Bn() | Relu()


def Fbt(*args, **kw): return Fl(*args, bias=False, **kw) | Bn() | Tanh()


def Fbs(*args, **kw): return Fl(*args, bias=False, **kw) | Bn() | Sig()


Textline2Img = wrapup(layers.Textline2Img)
Img2Seq = wrapup(layers.Img2Seq)
ImgMaxSeq = wrapup(layers.ImgMaxSeq)
ImgSumSeq = wrapup(layers.ImgSumSeq)
Gpu = wrapup(layers.Gpu)
Reorder = wrapup(layers.Reorder)
Permute = wrapup(layers.Permute)


def RowwiseLSTM(*args, **kw):
    def construct(arg):
        assert arg.rank() == 4
        return arg.add(layers.RowwiseLSTM(arg.size(1), *args, **kw))
    return Composable(construct)


def Lstm1(*args, **kw):
    def construct(arg):
        assert arg.rank() == 3
        return arg.add(layers.Lstm1(arg.size(1), *args, **kw))
    return Composable(construct)


def Lstm2(*args, **kw):
    def construct(arg):
        assert arg.rank() == 4
        return arg.add(layers.Lstm2(arg.size(1), *args, **kw))
    return Composable(construct)


def Lstm2to1(*args, **kw):
    def construct(arg):
        assert arg.rank() == 4
        # BDHW -> LBD
        return arg.add(layers.Lstm2to1(arg.size(1), *args, **kw))
    return Composable(construct)


def Lstm1to0(*args, **kw):
    def construct(arg):
        assert arg.rank() == 3
        # LBD -> BD
        return arg.add(layers.Lstm1to0(arg.size(1), *args, **kw))
    return Composable(construct)


def Lstm2to0(n1, n2=None):
    n2 = n2 or n1
    return Lstm2to1(n1) | Lstm1to0(N2)

