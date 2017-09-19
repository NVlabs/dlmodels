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


class Seq(nn.Sequential):
    def __init__(self, *modules):
        nn.Sequential.__init__(self)
        self._raw_modules = list(modules)
        self._input_shape = None
        self._output_shape = None

    def size(self):
        return self._output_shape

    def infer(self, shape):
        sample = Variable(torch.randn(*shape)) # FIXME
        shape = tuple(int(x) for x in sample.size())
        self._input_shape = shape
        for i, m in enumerate(self._raw_modules):
            if isinstance(m, (Inference, Seq)):
                m = m.infer(shape)
            sample = m.forward(sample)
            shape = tuple(sample.size())
            self.add_module("%d"%i, m)
        self._output_shape = tuple(int(x) for x in shape)
        return self

    def create(self, *args):
        self.infer(args)
        return self

    def __repr__(self):
        result = "%s -> %s\n" % (self._input_shape, self._output_shape)
        result += nn.Sequential.__repr__(self)
        return result
    def __str__(self):
        result = "%s -> %s\n" % (self._input_shape, self._output_shape)
        result += nn.Sequential.__str__(self)
        return result

    def __or__(self, other):
        if isinstance(other, Seq):
            new_list = self._raw_modules + other._raw_modules
        else:
            new_list = self._raw_modules + [other]
        return Seq(*new_list)

    def __pow__(self, n):
        new_list = self._raw_modules * n
        return Seq(*new_list)

Sig = nn.Sigmoid
Tanh = nn.Tanh
Relu = nn.ReLU
Smax = nn.Softmax
LSmax = nn.LogSoftmax
Reshape = layers.Viewer
PixelShuffle = nn.PixelShuffle

class Inference(object):
    def __init__(self, f):
        self.f = f
    def infer(self, shape):
        return self.f(shape)
    def create(self, *args):
        return self.infer(args)
    def __or__(self, other):
        return Seq(self, other)
    def __pow__(self, n):
        return Seq(*([self]*n))

def Bn(*args, **kw):
    def construct(shape):
        rank = len(shape)
        if rank == 5:
            return nn.BatchNorm3d(shape[1], *args, **kw)
        elif rank == 4:
            return nn.BatchNorm2d(shape[1], *args, **kw)
        elif rank == 3:
            return nn.BatchNorm1d(shape[1], *args, **kw)
        elif rank == 2:
            return nn.BatchNorm1d(shape[1], *args, **kw)
        raise Exception("%d: bad rank for Bn")
    return Inference(construct)


def Mp(*args, **kw):
    def construct(shape):
        rank = len(shape)
        if rank == 5:
            return nn.MaxPool3d(*args, **kw)
        elif rank == 4:
            return nn.MaxPool2d(*args, **kw)
        elif rank == 3:
            return nn.MaxPool1d(*args, **kw)
        raise Exception("%d: bad rank for Cl")
    return Inference(construct)


def Ap(*args, **kw):
    def construct(shape):
        rank = len(shape)
        if rank == 5:
            return nn.AvgPool3d(*args, **kw)
        elif rank == 4:
            return nn.AvgPool2d(*args, **kw)
        elif rank == 3:
            return nn.AvgPool1d(*args, **kw)
        raise Exception("%d: bad rank for Cl")
    return Inference(construct)


def Cl(out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=True):
    def construct(shape):
        ksize = kernel_size
        pad = padding
        rank = len(shape)
        assert rank > 2
        if isinstance(ksize, int):
            ksize = tuple([ksize] * (rank - 2))
        if pad is None:
            pad = tuple([x // 2 for x in ksize])
        rank = len(shape)
        if rank == 5:
            return nn.Conv3d(shape[1], out_channels, ksize,
                                 stride, pad, dilation, groups, bias)
        elif rank == 4:
            return nn.Conv2d(shape[1], out_channels, ksize,
                                 stride, pad, dilation, groups, bias)
        elif rank == 3:
            return nn.Conv1d(shape[1], out_channels, ksize,
                                 stride, pad, dilation, groups, bias)
        raise Exception("%d: bad rank for Cl")
    return Inference(construct)


def Cr(*args, **kw): return Seq(Cl(*args, **kw), Relu())


def Ct(*args, **kw): return Seq(Cl(*args, **kw), Tanh())


def Cs(*args, **kw): return Seq(Cl(*args, **kw), Sig())


def Cbl(*args, **kw): return Seq(Cl(*args, bias=False, **kw), Bn())


def Cbr(*args, **kw): return Seq(Cl(*args, bias=False, **kw), Bn(), Relu())


def Cbt(*args, **kw): return Seq(Cl(*args, bias=False, **kw), Bn(), Tanh())


def Cbs(*args, **kw): return Seq(Cl(*args, bias=False, **kw), Bn(), Sig())


def Dws(n, **kw): return Cl(n, 1, **kw)


def Flat():
    def construct(shape):
        rank = len(shape)
        assert rank > 2
        new_depth = np.prod(shape[1:])
        return layers.Viewer(-1, new_depth)
    return Inference(construct)


def Fl(*args, **kw):
    def construct(shape):
        assert len(shape) == 2
        return nn.Linear(shape[1], *args, **kw)
    return Inference(construct)


def Fr(*args, **kw): return Seq(Fl(*args, **kw), Relu())


def Ft(*args, **kw): return Seq(Fl(*args, **kw), Tanh())


def Fs(*args, **kw): return Seq(Fl(*args, **kw), Sig())


def Fbl(*args, **kw): return Seq(Fl(*args, bias=False, **kw), Bn())


def Fbr(*args, **kw): return Seq(Fl(*args, bias=False, **kw), Bn(), Relu())


def Fbt(*args, **kw): return Seq(Fl(*args, bias=False, **kw), Bn(), Tanh())


def Fbs(*args, **kw): return Seq(Fl(*args, bias=False, **kw), Bn(), Sig())


Textline2Img = layers.Textline2Img
Img2Seq = layers.Img2Seq
ImgMaxSeq = layers.ImgMaxSeq
ImgSumSeq = layers.ImgSumSeq
Reorder = layers.Reorder
Permute = layers.Permute


def Gpu():
    def construct(shape):
        return layers.Gpu()
    return Inference(construct)

def Check():
    def construct(shape):
        return layers.Check()
    return Inference(construct)

def Info():
    def construct(shape):
        return layers.Info()
    return Inference(construct)

def RowwiseLSTM(*args, **kw):
    def construct(shape):
        assert len(shape) == 4
        return layers.RowwiseLSTM(shape[1], *args, **kw)
    return Inference(construct)


def Lstm1(*args, **kw):
    def construct(shape):
        assert len(shape) == 3
        return layers.Lstm1(shape[1], *args, **kw)
    return Inference(construct)


def Lstm2(*args, **kw):
    def construct(shape):
        assert len(shape) == 4
        return layers.Lstm2(shape[1], *args, **kw)
    return Inference(construct)


def Lstm2to1(*args, **kw):
    def construct(shape):
        assert len(shape) == 4
        # BDHW -> LBD
        return layers.Lstm2to1(shape[1], *args, **kw)
    return Inference(construct)


def Lstm1to0(*args, **kw):
    def construct(shape):
        assert len(shape) == 3
        # LBD -> BD
        return layers.Lstm1to0(shape[1], *args, **kw)
    return Inference(construct)


def Lstm2to0(n1, n2=None):
    n2 = n2 or n1
    return Seq(Lstm2to1(n1), Lstm1to0(N2))
