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
    ck.verify(x + y, {x: tvm.arith.IntervalSet(0, 10)}, (y, 10 + y))
    ck.verify(x + y, {x: tvm.arith.IntervalSet(0, 10), y: tvm.arith.IntervalSet(1, 11)}, (1, 21))
    ck.verify(x - y, {x: tvm.arith.IntervalSet(0, 10), y: tvm.arith.IntervalSet(1, 11)}, (-11, 9))


def test_mul_div():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")

    tdiv = tvm.tir.truncdiv
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(x * y, {x: tvm.arith.IntervalSet(0, 10)}, (0, 10 * y))
    ck.verify(x * 2, {x: tvm.arith.IntervalSet(1, 10)}, (2, 20))
    ck.verify(x * -2, {x: tvm.arith.IntervalSet(1, 10)}, (-20, -2))

    ck.verify(tdiv(x, y), {x: tvm.arith.IntervalSet(0, 10)}, (0, tdiv(10, y)))
    ck.verify(tdiv(x, 2), {x: tvm.arith.IntervalSet(1, 10)}, (0, 5))

    fld = tvm.te.floordiv
    ck.verify(fld(x, y), {x: tvm.arith.IntervalSet(0, 10)}, (0, fld(10, y)))
    ck.verify(fld(x, 2), {x: tvm.arith.IntervalSet(-1, 10)}, (-1, 5))


def test_mod():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    tmod = tvm.tir.truncmod
    ck.analyzer.update(y, tvm.arith.ConstIntBound(1, 100), override=True)
    ck.verify(tmod(x, y), {x: tvm.arith.IntervalSet(0, 10)}, (0, y - 1))
    ck.verify(tmod(x, 10), {x: tvm.arith.IntervalSet(1, 10)}, (0, 9))

    flm = tvm.te.floormod
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(-10, 10)}, (0, 9))
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(3, 5)}, (3, 5))
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(13, 15)}, (3, 5))
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(3, 15)}, (0, 9))
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(3, 11)}, (0, 9))
    ck.verify(flm(x, 10), {x: tvm.arith.IntervalSet(1, 21)}, (0, 9))

    floordiv = tvm.te.floordiv
    z = te.var("z")
    ck.analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 3))
    ck.verify(flm(y, 8), {y: tvm.arith.IntervalSet(z * 8 + x * 4, z * 8 + x * 4 + 3)}, (0, 7))
    ck1 = IntSetChecker()
    ck1.analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 2))
    ck1.verify(
        flm(y, 8), {y: tvm.arith.IntervalSet(z * 8 + x * 4, z * 8 + x * 4 + 3)}, (x * 4, x * 4 + 3)
    )


def test_max_min():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    ck.verify(tvm.te.max(x, x + 1), {x: tvm.arith.IntervalSet(0, 10)}, (1, 11))
    ck.verify(tvm.te.min(x - 1, x + 1), {x: tvm.arith.IntervalSet(0, 10)}, (-1, 9))
    ck.verify(tvm.te.min(x, y), {}, (tvm.te.min(x, y), tvm.te.min(x, y)))
    ck.verify(tvm.te.max(x, y), {}, (tvm.te.max(x, y), tvm.te.max(x, y)))


def test_select():
    ck = IntSetChecker()
    x, y = te.var("x"), te.var("y")
    ck.verify(tvm.tir.Select(x > 0, x - 1, x + 1), {x: tvm.arith.IntervalSet(0, 10)}, (-1, 11))


def test_region_lower_bound_not_independent():
    i = tvm.tir.Var("i", "int32")
    result = tvm.arith.estimate_region_lower_bound(
        region=[
            tvm.ir.Range(begin=i, end=i + 2),
            tvm.ir.Range(begin=i + 1, end=i + 4),
        ],
        var_dom={
            i: tvm.ir.Range(begin=0, end=64),
        },
        predicate=tvm.tir.IntImm("bool", 1),
    )
    assert result is None


def test_region_lower_bound_stride_too_wide():
    i = tvm.tir.Var("i", "int32")
    result = tvm.arith.estimate_region_lower_bound(
        region=[
            tvm.ir.Range(begin=i * 4, end=i * 4 + 2),
        ],
        var_dom={
            i: tvm.ir.Range(begin=0, end=64),
        },
        predicate=tvm.tir.IntImm("bool", 1),
    )
    assert result is None


def test_region_lower_bound_small_stride():
    i = tvm.tir.Var("i", "int32")
    (result,) = tvm.arith.estimate_region_lower_bound(
        region=[
            tvm.ir.Range.from_min_extent(min_value=i * 4, extent=8),
        ],
        var_dom={
            i: tvm.ir.Range(begin=0, end=64),
        },
        predicate=tvm.tir.IntImm("bool", 1),
    )
    assert result.min_value.value == 0
    assert result.max_value.value == 259


def test_region_lower_bound_split_predicate():
    x_o = tvm.tir.Var("xo", "int32")
    x_i = tvm.tir.Var("xi", "int32")
    x = x_o * 4 + x_i
    (result,) = tvm.arith.estimate_region_lower_bound(
        region=[
            tvm.ir.Range.from_min_extent(min_value=x * 4, extent=8),
        ],
        var_dom={
            x_o: tvm.ir.Range(begin=0, end=16),
            x_i: tvm.ir.Range(begin=0, end=4),
        },
        predicate=x < 63,
    )
    assert result.min_value.value == 0
    assert result.max_value.value == 255


def test_region_lower_bound_multiple_variables():
    div = tvm.tir.floordiv
    mod = tvm.tir.floormod
    x = tvm.tir.Var("x", "int32")
    wid = tvm.tir.Var("wid", "int32")
    i = div(x, 16)
    j = div(mod(x, 16), 4) * 8 + mod(x, 4) + div(wid, 32) * 4
    k = wid % 32
    (i_int_set, j_int_set, k_int_set) = tvm.arith.estimate_region_lower_bound(
        region=[
            tvm.ir.Range.from_min_extent(min_value=i, extent=1),
            tvm.ir.Range.from_min_extent(min_value=j, extent=1),
            tvm.ir.Range.from_min_extent(min_value=k, extent=1),
        ],
        var_dom={
            x: tvm.ir.Range(begin=0, end=32),
            wid: tvm.ir.Range(begin=0, end=64),
        },
        predicate=tvm.tir.IntImm("bool", 1),
    )
    assert i_int_set.min_value.value == 0
    assert i_int_set.max_value.value == 1
    assert j_int_set.min_value.value == 0
    assert j_int_set.max_value.value == 31
    assert k_int_set.min_value.value == 0
    assert k_int_set.max_value.value == 31


def test_union_lower_bound():
    neg_inf = tvm.arith.int_set.neg_inf()
    pos_inf = tvm.arith.int_set.pos_inf()
    set_0 = tvm.arith.IntervalSet(min_value=neg_inf, max_value=0)
    set_1 = tvm.arith.IntervalSet(min_value=1, max_value=pos_inf)
    result = tvm.arith.int_set.union_lower_bound([set_0, set_1])
    assert result.min_value.same_as(neg_inf)
    assert result.max_value.same_as(pos_inf)


if __name__ == "__main__":
    test_basic()
    test_vector()
    test_add_sub()
    test_mul_div()
    test_max_min()
    test_select()
    test_mod()
    test_region_lower_bound_not_independent()
    test_region_lower_bound_stride_too_wide()
    test_region_lower_bound_small_stride()
    test_region_lower_bound_split_predicate()
    test_region_lower_bound_multiple_variables()
    test_union_lower_bound()
