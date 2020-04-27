# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import te


class IntSetChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, dmap, expected):
        res = self.analyzer.int_set(data, dmap)
        def err_msg():
            return "\ndata={}\ndmap={}\nres={}\nexpected={}".format(data, dmap, res, expected)
        def equal(x, y):
            res = self.analyzer.canonical_simplify(x - y)
            return tvm.tir.analysis.expr_deep_equal(res, 0)
        assert equal(res.min_value, expected[0]), err_msg()
        assert equal(res.max_value, expected[1]), err_msg()

def test_basic():
    s = tvm.arith.IntervalSet(2, 3)
    assert s.min_value.value == 2
    assert s.max_value.value == 3

    s = tvm.arith.IntSet.single_point(2)
    assert s.min_value.value == 2
    assert s.max_value.value == 2


def test_vector():
    base = 10
    stride = 3
    lanes = 2
    s = tvm.arith.IntSet.vector(tvm.tir.Ramp(base, stride, lanes))
    assert s.min_value.value == base
    assert s.max_value.value == base + stride * lanes - 1


def test_add_sub():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    ck.verify(x + y, {x : tvm.arith.IntervalSet(0, 10)}, (y, 10 + y))
    ck.verify(x + y,
              {x : tvm.arith.IntervalSet(0, 10), y : tvm.arith.IntervalSet(1, 11)},
              (1, 21))
    ck.verify(x - y,
              {x : tvm.arith.IntervalSet(0, 10), y : tvm.arith.IntervalSet(1, 11)},
              (-11, 9))

def test_mul_div():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")

    tdiv = tvm.tir.truncdiv
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(x * y, {x : tvm.arith.IntervalSet(0, 10)}, (0, 10 * y))
    ck.verify(x * 2, {x : tvm.arith.IntervalSet(1, 10)}, (2, 20))
    ck.verify(x * -2, {x : tvm.arith.IntervalSet(1, 10)}, (-20, -2))

    ck.verify(tdiv(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, tdiv(10, y)))
    ck.verify(tdiv(x, 2), {x : tvm.arith.IntervalSet(1, 10)}, (0, 5))

    fld = tvm.te.floordiv
    ck.verify(fld(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, fld(10, y)))
    ck.verify(fld(x, 2), {x : tvm.arith.IntervalSet(-1, 10)}, (-1, 5))


def test_mod():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    tmod = tvm.tir.truncmod
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(tmod(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, y - 1))
    ck.verify(tmod(x, 10), {x : tvm.arith.IntervalSet(1, 10)}, (0, 9))

    flm = tvm.te.floormod
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(-10, 10)}, (0, 9))
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(3, 5)}, (3, 5))
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(13, 15)}, (3, 5))
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(3, 15)}, (0, 9))
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(3, 11)}, (0, 9))
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(1, 21)}, (0, 9))

    floordiv = tvm.te.floordiv
    z = te.var("z")
    ck.analyzer.bind(x, tvm.ir.Range.make_by_min_extent(0, 3))
    ck.verify(flm(y, 8), {y : tvm.arith.IntervalSet(z*8+x*4, z*8+x*4+3)},
              (0, 7))
    ck1 = IntSetChecker()
    ck1.analyzer.bind(x, tvm.ir.Range.make_by_min_extent(0, 2))
    ck1.verify(flm(y, 8), {y : tvm.arith.IntervalSet(z*8+x*4, z*8+x*4+3)}, (x*4, x*4+3))


def test_max_min():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    ck.verify(tvm.te.max(x, x + 1), {x : tvm.arith.IntervalSet(0, 10)}, (1, 11))
    ck.verify(tvm.te.min(x - 1, x + 1), {x : tvm.arith.IntervalSet(0, 10)}, (-1, 9))
    ck.verify(tvm.te.min(x, y), {}, (tvm.te.min(x, y), tvm.te.min(x, y)))
    ck.verify(tvm.te.max(x, y), {}, (tvm.te.max(x, y), tvm.te.max(x, y)))


def test_select():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    ck.verify(tvm.tir.Select(x > 0, x - 1, x + 1),
              {x : tvm.arith.IntervalSet(0, 10)}, (-1, 11))


if __name__ == "__main__":
    test_basic()
    test_vector()
    test_add_sub()
    test_mul_div()
    test_max_min()
    test_select()
    test_mod()
