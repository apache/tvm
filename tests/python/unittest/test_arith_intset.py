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


class IntSetChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, dmap, expected):
        res = self.analyzer.int_set(data, dmap)
        def err_msg():
            return "\ndata={}\ndmap={}\nres={}\nexpected={}".format(data, dmap, res, expected)
        def equal(x, y):
            res = self.analyzer.canonical_simplify(x - y)
            return tvm.ir_pass.Equal(res, 0)
        assert equal(res.min_value, expected[0]), err_msg()
        assert equal(res.max_value, expected[1]), err_msg()

def test_basic():
    s = tvm.arith.IntervalSet(2, 3)
    assert s.min_value.value == 2
    assert s.max_value.value == 3


def test_vector():
    base = 10
    stride = 3
    lanes = 2
    s = tvm.arith.intset_vector(tvm.make.Ramp(base, stride, lanes))
    assert s.min_value.value == base
    assert s.max_value.value == base + stride * lanes - 1


def test_add_sub():
    ck = IntSetChecker()
    x, y = tvm.var("x"), tvm.var("y")
    ck.verify(x + y, {x : tvm.arith.IntervalSet(0, 10)}, (y, 10 + y))
    ck.verify(x + y,
              {x : tvm.arith.IntervalSet(0, 10), y : tvm.arith.IntervalSet(1, 11)},
              (1, 21))
    ck.verify(x - y,
              {x : tvm.arith.IntervalSet(0, 10), y : tvm.arith.IntervalSet(1, 11)},
              (-11, 9))

def test_mul_div():
    ck = IntSetChecker()
    x, y = tvm.var("x"), tvm.var("y")
    sx, sy = tvm.shape_var("sx"), tvm.shape_var("sy")

    tdiv = tvm.truncdiv
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(x * y, {x : tvm.arith.IntervalSet(0, 10)}, (0, 10 * y))
    ck.verify(x * 2, {x : tvm.arith.IntervalSet(1, 10)}, (2, 20))
    ck.verify(x * -2, {x : tvm.arith.IntervalSet(1, 10)}, (-20, -2))

    ck.verify(tdiv(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, tdiv(10, y)))
    ck.verify(tdiv(x, 2), {x : tvm.arith.IntervalSet(1, 10)}, (0, 5))
    ck.verify(tdiv(sx, 2), {}, (0,  tdiv(sx, 2)))
    ck.verify(tdiv(2, sy), {}, (tdiv(2, sy),  tdiv(2, sy)))
    ck.verify(tdiv(sx, sy), {}, (tdiv(sx, sy),  tdiv(sx, sy)))

    fld = tvm.floordiv
    ck.verify(fld(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, fld(10, y)))
    ck.verify(fld(x, 2), {x : tvm.arith.IntervalSet(-1, 10)}, (-1, 5))
    ck.verify(fld(sx, 2), {}, (0,  fld(sx, 2)))
    ck.verify(fld(2, sy), {}, (fld(2, sy),  fld(2, sy)))
    ck.verify(fld(sx, sy), {}, (fld(sx, sy),  fld(sx, sy)))


def test_mod():
    ck = IntSetChecker()
    x, y = tvm.var("x"), tvm.var("y")
    sx, sy = tvm.shape_var("sx"), tvm.shape_var("sy")
    tmod = tvm.truncmod
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(tmod(x, y), {x : tvm.arith.IntervalSet(0, 10)}, (0, y - 1))
    ck.verify(tmod(x, 10), {x : tvm.arith.IntervalSet(1, 10)}, (0, 9))
    ck.verify(tmod(sx, 2), {}, (0, 1))
    ck.verify(tmod(2, sy), {}, (tmod(2, sy),  tmod(2, sy)))
    ck.verify(tmod(sx, sy), {}, (tmod(sx, sy),  tmod(sx, sy)))

    flm = tvm.floormod
    ck.verify(flm(x, 10), {x : tvm.arith.IntervalSet(-10, 10)}, (0, 9))
    ck.verify(flm(sx, 2), {}, (0, 1))
    ck.verify(flm(2, sy), {}, (flm(2, sy),  flm(2, sy)))
    ck.verify(flm(sx, sy), {}, (flm(sx, sy),  flm(sx, sy)))


def test_max_min():
    ck = IntSetChecker()
    x, y = tvm.var("x"), tvm.var("y")
    ck.verify(tvm.max(x, x + 1), {x : tvm.arith.IntervalSet(0, 10)}, (1, 11))
    ck.verify(tvm.min(x - 1, x + 1), {x : tvm.arith.IntervalSet(0, 10)}, (-1, 9))
    ck.verify(tvm.min(x, y), {}, (tvm.min(x, y), tvm.min(x, y)))
    ck.verify(tvm.max(x, y), {}, (tvm.max(x, y), tvm.max(x, y)))


def test_select():
    ck = IntSetChecker()
    x, y = tvm.var("x"), tvm.var("y")
    ck.verify(tvm.expr.Select(x > 0, x - 1, x + 1),
              {x : tvm.arith.IntervalSet(0, 10)}, (-1, 11))


if __name__ == "__main__":
    test_basic()
    test_vector()
    test_add_sub()
    test_mul_div()
    test_max_min()
    test_select()
    test_mod()
