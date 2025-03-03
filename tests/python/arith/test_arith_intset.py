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
import tvm.testing
from tvm import te
from tvm import tir
from tvm.arith.analyzer import Analyzer


class IntSetChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, dmap, expected):
        res = self.analyzer.int_set(data, dmap)

        def err_msg():
            return "\ndata={}\ndmap={}\nres={}\nexpected={}".format(data, dmap, res, expected)

        assert self.analyzer.can_prove_equal(res.min_value, expected[0]), err_msg()
        assert self.analyzer.can_prove_equal(res.max_value, expected[1]), err_msg()


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
    assert s.max_value.value == base + stride * (lanes - 1)


def test_scalable_vector():
    base = 5
    s = tvm.arith.IntSet.vector(tvm.tir.Ramp(base, 2, tvm.tir.vscale() * 4))

    assert s.min_value.value == base
    assert s.max_value.same_as(tvm.arith.int_set.pos_inf())


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

    fld = tvm.te.floordiv
    z = te.var("z")
    ck.analyzer.bind(x, tvm.ir.Range.from_min_extent(0, 3))
    ck.verify(
        flm(y, 8),
        {y: tvm.arith.IntervalSet(z * 8 + x * 4, z * 8 + x * 4 + 3)},
        (
            z * 8 + x * 4 - 8 * fld(z * 8 + x * 4, 8),
            z * 8 + x * 4 + 3 - 8 * fld(z * 8 + x * 4, 8),
        ),
    )
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


def check_region_bound(expect_region, var_dom, mode, predicate=None):
    """Helper to check region bound estimation.

    Parameters
    ----------
    expect_region: dict
        The keys are of form (begin, end) or PrimExpr as a single point. The values are
        expected estimated region or region dict on different bindings.

    var_dom: dict
        Map var to iteration domain range.

    mode: str
        Specify "lowerbound", "upperbound" or else use strict bound estimation.

    predicate: PrimExpr
        Extra predicate, defaults to True.
    """
    if predicate is None:
        predicate = tvm.tir.IntImm("bool", 1)
    region = []
    expect = []
    for k, v in expect_region.items():
        if not isinstance(k, (tuple, list)):
            k = (k, k + 1)
        region.append(tvm.ir.Range.from_min_extent(k[0], Analyzer().simplify(k[1] - k[0])))
        expect.append(v)
    if mode == "lowerbound":
        result = tvm.arith.estimate_region_lower_bound(
            region=region, var_dom=var_dom, predicate=predicate
        )
    elif mode == "upperbound":
        result = tvm.arith.estimate_region_upper_bound(
            region=region, var_dom=var_dom, predicate=predicate
        )
    else:
        result = tvm.arith.estimate_region_strict_bound(
            region=region, var_dom=var_dom, predicate=predicate
        )
    if result is None:
        assert all([_ is None for _ in expect])
        return
    assert len(result) == len(expect)
    for intset, expect_desc in zip(result, expect):
        if isinstance(expect_desc, dict):
            # check range on different free var bindings
            for binding in expect_desc:
                analyzer = Analyzer()
                for k, v in binding:
                    analyzer.bind(k, v)
                expect_begin, expect_end = expect_desc[binding]
                result_begin = analyzer.simplify(intset.min_value, 3)
                result_end = analyzer.simplify(intset.max_value + 1, 3)
                assert analyzer.can_prove_equal(
                    result_begin - expect_begin, 0
                ), f"{result_begin} vs {expect_begin}"
                assert analyzer.can_prove_equal(
                    result_end - expect_end, 0
                ), f"{result_end} vs {expect_end}"
        else:
            # check range
            expect_begin, expect_end = expect_desc
            analyzer = Analyzer()
            assert analyzer.can_prove_equal(
                intset.min_value - expect_begin, 0
            ), f"{intset.min_value} vs {expect_begin}"
            assert analyzer.can_prove_equal(
                intset.max_value - expect_end + 1, 0
            ), f"{intset.max_value} vs {expect_end - 1}"


def test_region_bound_not_independent():
    # (i, i+2) and (i+2, i+4) are dependent, this the lowerbound is not available
    i = tvm.tir.Var("i", "int32")
    var_dom = {
        i: tvm.ir.Range(begin=0, end=64),
    }
    check_region_bound({(i, i + 2): None, (i + 2, i + 4): None}, var_dom, mode="lowerbound")
    check_region_bound({(i, i + 2): (0, 65), (i + 2, i + 4): (2, 67)}, var_dom, mode="upperbound")

    # when only a subset of access indices are affine
    i, j, k = tvm.tir.Var("i", "int32"), tvm.tir.Var("j", "int32"), tvm.tir.Var("k", "int32")
    var_dom = {
        i: tvm.ir.Range(begin=0, end=16),
        j: tvm.ir.Range(begin=0, end=16),
        k: tvm.ir.Range(begin=0, end=16),
    }
    check_region_bound(
        {i // 4: None, j * 4 + i % 4: None, tir.truncdiv(k, 2): None},
        var_dom,
        predicate=j * 4 + i % 4 > 3,
        mode="lowerbound",
    )
    check_region_bound(
        {i // 4: (0, 4), j * 4 + i % 4: (4, 64), tir.truncdiv(k, 2): (0, 8)},
        var_dom,
        predicate=j * 4 + i % 4 > 3,
        mode="upperbound",
    )


def test_region_bound_stride_too_wide():
    i = tvm.tir.Var("i", "int32")
    var_dom = {i: tvm.ir.Range(begin=0, end=64)}
    check_region_bound({(i * 4, i * 4 + 2): None}, var_dom, mode="lowerbound")
    check_region_bound({(i * 4, i * 4 + 2): (0, 254)}, var_dom, mode="upperbound")


def test_region_bound_small_stride():
    i = tvm.tir.Var("i", "int32")
    var_dom = {
        i: tvm.ir.Range(begin=0, end=64),
    }
    check_region_bound({(i * 4, i * 4 + 8): (0, 260)}, var_dom, mode="lowerbound")


def test_region_lower_bound_split_predicate():
    x_o = tvm.tir.Var("xo", "int32")
    x_i = tvm.tir.Var("xi", "int32")
    x = x_o * 4 + x_i
    var_dom = {
        x_o: tvm.ir.Range(begin=0, end=16),
        x_i: tvm.ir.Range(begin=0, end=4),
    }
    check_region_bound({(x * 4, x * 4 + 8): (0, 256)}, var_dom, predicate=x < 63, mode="lowerbound")

    check_region_bound(
        {(x * 4, x * 4 + 8): (0, 256), (x * 3, x * 3 + 5): (0, 191)},
        var_dom,
        predicate=x < 63,
        mode="upperbound",
    )


def test_region_lower_bound_multiple_variables():
    div = tvm.tir.floordiv
    mod = tvm.tir.floormod
    x = tvm.tir.Var("x", "int32")
    wid = tvm.tir.Var("wid", "int32")
    i = div(x, 16)
    j = div(mod(x, 16), 4) * 8 + mod(x, 4) + div(wid, 32) * 4
    k = wid % 32
    var_dom = {
        x: tvm.ir.Range(begin=0, end=32),
        wid: tvm.ir.Range(begin=0, end=64),
    }
    check_region_bound({i: (0, 2), j: (0, 32), k: (0, 32)}, var_dom, mode="lowerbound")


def test_region_lower_bound_negative_scale():
    i = tvm.tir.Var("i", "int32")
    j = tvm.tir.Var("j", "int32")
    var_dom = {
        i: tvm.ir.Range(begin=0, end=4),
        j: tvm.ir.Range(begin=0, end=4),
    }
    check_region_bound(
        {(1 - i, 5 - i): (-2, 5), (20 - j * 4, 36 - j * 4): (8, 36)}, var_dom, mode="lowerbound"
    )


def test_region_lower_bound_for_non_perfect_tile():
    h1 = tvm.tir.Var("h1", "int32")
    h2 = tvm.tir.Var("h2", "int32")
    h3 = tvm.tir.Var("h3", "int32")

    # non-uniform tiling, single inner variable
    var_dom = {
        h2: tvm.ir.Range(begin=0, end=10),
    }
    check_region_bound(
        {
            h3 * 8
            + h2: {
                (): (
                    tvm.tir.max(h3 * 8, 1),
                    tvm.tir.min(0, h3 * 8 - 214) + 224,
                ),
                ((h3, 0),): (1, 10),  # h3 == 0: region is [1, 10)
                ((h3, 10),): (h3 * 8, h3 * 8 + 10),  # 0 < h3 <= 26: region is [h3 * 8, h3 * 8 + 10)
                ((h3, 27),): (h3 * 8, 224),  # h3 > 26: region is [h3 * 8, 224)
            }
        },
        var_dom,
        predicate=tvm.tir.all(1 <= h3 * 8 + h2, h3 * 8 + h2 < 224),
        mode="lowerbound",
    )

    # non-uniform tiling, two inner variables
    var_dom = {
        h1: tvm.ir.Range(begin=0, end=5),
        h2: tvm.ir.Range(begin=0, end=2),
    }
    check_region_bound(
        {
            h3 * 8
            + h2 * 5
            + h1: {
                (): (
                    tvm.tir.max(h3 * 8, 1),
                    tvm.tir.min(0, h3 * 8 - 214) + 224,
                ),
                ((h3, 0),): (1, 10),
                ((h3, 10),): (h3 * 8, h3 * 8 + 10),
                ((h3, 27),): (h3 * 8, 224),
            }
        },
        var_dom,
        predicate=tvm.tir.all(1 <= h3 * 8 + h2 * 5 + h1, h3 * 8 + h2 * 5 + h1 < 224),
        mode="lowerbound",
    )

    # lowerbound should fail on incompatible predicates
    check_region_bound(
        {h3 * 8 + h2 * 5 + h1: None},
        var_dom,
        predicate=tvm.tir.all(1 <= h3 * 8 + h2 * 5 + h1, h3 * 8 + h1 * 2 + h2 < 224),
        mode="lowerbound",
    )
    check_region_bound(
        {h3 * 8 + h2 * 5 + h1: (h3 * 8, h3 * 8 + 10)},
        var_dom,
        predicate=tvm.tir.all(1 <= h3 * 8 + h2 * 5 + h1, h3 * 8 + h1 * 2 + h2 < 224),
        mode="upperbound",
    )


def test_region_lower_bound_unfusable():
    var_dom = {
        tvm.tir.Var("i", "int32"): tvm.ir.Range(8),
        tvm.tir.Var("j", "int32"): tvm.ir.Range(4),
    }
    i, j = var_dom
    check_region_bound({(i + j) // 2: (0, 6)}, var_dom, mode="lowerbound")


def test_union_lower_bound():
    neg_inf = tvm.arith.int_set.neg_inf()
    pos_inf = tvm.arith.int_set.pos_inf()
    set_0 = tvm.arith.IntervalSet(min_value=neg_inf, max_value=0)
    set_1 = tvm.arith.IntervalSet(min_value=1, max_value=pos_inf)
    result = tvm.arith.int_set.union_lower_bound([set_0, set_1])
    assert result.min_value.same_as(neg_inf)
    assert result.max_value.same_as(pos_inf)
    set_2 = tvm.arith.IntervalSet(min_value=pos_inf, max_value=neg_inf)
    result = tvm.arith.int_set.union_lower_bound([set_0, set_1, set_2])
    assert result.min_value.same_as(neg_inf)
    assert result.max_value.same_as(pos_inf)


if __name__ == "__main__":
    tvm.testing.main()
