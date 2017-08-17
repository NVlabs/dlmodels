# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).

from dlmodels.specs import *

p = Cr(64, 3) | Mp(2, 2) | Cr(128, 3) | Mp(2, 2)
m = p.create(17, 4, 64, 64)
assert m.size() == (17, 128, 16, 16)

p = (Cr(64, 3) | Mp(2, 2))**3
m = p.create(17, 4, 64, 64)
assert m.size() == (17, 64, 8, 8)

p = Lstm2(20, ndir=1)
m = p.create(17, 4, 100, 64)
assert m.size() == (17, 20, 100, 64), m

p = Lstm2(20)
m = p.create(17, 4, 100, 64)
assert m.size() == (17, 40, 100, 64), m

p = Lstm1(20)
m = p.create(17, 4, 100)
assert m.size() == (17, 40, 100), m

p = Lstm2to1(20)
m = p.create(17, 4, 100, 64)
assert m.size() == (17, 20, 100), m

p = Lstm1to0(20)
m = p.create(17, 4, 100)
assert m.size() == (17, 20), m

p = Cr(64, 3) | Mp(2, 2) | Cr(128, 3) | Mp(2, 2)
m = p.create(17, 4, 64, 64)
assert m.size() == (17, 128, 16, 16)

p = Cr(64, 3) | PixelShuffle(4)
m = p.create(17, 3, 8, 8)
assert m.size() == (17, 4, 32, 32), m.size()
