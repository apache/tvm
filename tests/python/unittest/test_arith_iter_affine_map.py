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


def ifuse(inputs):
    """Fuse iterators"""
    value, extent = 0, 1
    for i, ext in inputs:
        value = value * ext + i
        extent = extent * ext
    return (value, extent)


def isplit(axis, factor):
    """Split iterators"""
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod
    return [
        (fld(axis[0], factor), fld(axis[1] + (factor - 1), factor)),
        (flm(axis[0], factor), factor),
    ]


def var_dom(iters):
    """Get domains of iterators"""
    return {var: tvm.ir.Range(0, ext) for var, ext in iters}


def assert_iter_sum_pattern(sum_expr, extent, base, scale=1):
    """Check the sum expr have the right pattern."""
    assert isinstance(sum_expr, tvm.arith.IterSumExpr)
    if extent == 1:
        assert len(sum_expr.args) == 0
    else:
        assert len(sum_expr.args) == 1
        tvm.testing.assert_prim_expr_equal(sum_expr.args[0].extent, extent)
        tvm.testing.assert_prim_expr_equal(sum_expr.args[0].scale, scale)
    tvm.testing.assert_prim_expr_equal(sum_expr.base, base)


def test_trivial():
    x = tvm.tir.Var("x", "int32"), 3
    y = tvm.tir.Var("y", "int32"), 4

    res = tvm.arith.detect_iter_map([x[0], y[0], 3], var_dom([x, y]))

    assert len(res) == 3
    assert_iter_sum_pattern(res[0], 3, 0)
    assert_iter_sum_pattern(res[1], 4, 0)
    assert_iter_sum_pattern(res[2], 1, 3)

    res = tvm.arith.detect_iter_map([x[0], 3], var_dom([x, y]))
    assert len(res) == 0

    # not independent
    res = tvm.arith.detect_iter_map([x[0], x[0], 3], var_dom([x, y]))
    assert len(res) == 0


def test_fuse():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    c = tvm.tir.SizeVar("c", "int32")
    c0 = tvm.tir.SizeVar("c0", "int32")
    c1 = tvm.tir.SizeVar("c1", "int32")
    c2 = tvm.tir.SizeVar("c1", "int32")

    res = tvm.arith.detect_iter_map([y * 3 + 1 + c + x], var_dom([(x, 3), (y, 4)]))
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 12, 1 + c)

    res = tvm.arith.detect_iter_map([ifuse([(x, 3), (y, 4)])[0]], var_dom([(x, 3), (y, 4)]))
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 12, 0)

    # fuse with symbolic factor
    res = tvm.arith.detect_iter_map([(y + 1) * c + x], var_dom([(x, c), (y, 4)]))
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 4 * c, c)

    # duplication
    res = tvm.arith.detect_iter_map([y * 3 + x, y], var_dom([(x, 3), (y, 4)]))
    assert len(res) == 0

    # duplication 2
    res = tvm.arith.detect_iter_map([y, x + 1, y], var_dom([(x, 3), (y, 4)]))
    assert len(res) == 0

    # factor mismatch
    res = tvm.arith.detect_iter_map([y * 4 + x], var_dom([(x, 3), (y, 4)]))
    assert len(res) == 0

    # simple stride pattern
    res = tvm.arith.detect_iter_map([x * 4 + y * 2], var_dom([(x, 3), (y, 2)]))
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 6, 0, scale=2)

    # simple stride pattern with symbolic
    res = tvm.arith.detect_iter_map([x * 2 * c0 + y * 2], var_dom([(x, 3), (y, c0)]))
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 3 * c0, 0, scale=2)


def test_split():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("y", "int32")
    c0 = tvm.tir.SizeVar("c0", "int32")
    c1 = tvm.tir.SizeVar("c1", "int32")
    c2 = tvm.tir.SizeVar("c1", "int32")
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod

    res = tvm.arith.detect_iter_map([fld(x, 3), flm(x, 3) * 2 + c1], var_dom([(x, 24)]))

    assert len(res) == 2
    assert_iter_sum_pattern(res[0], 8, 0)
    assert_iter_sum_pattern(res[1], 3, c1, 2)

    res = tvm.arith.detect_iter_map([fld(x, 6), fld(flm(x, 6), 2), flm(x, 2)], var_dom([(x, 24)]))

    assert len(res) == 3
    assert_iter_sum_pattern(res[0], 4, 0)
    assert_iter_sum_pattern(res[1], 3, 0)
    assert_iter_sum_pattern(res[2], 2, 0)

    # simple symbolic bound
    # TODO(tvm-team) improve symbolic divisible check to enable
    # more complicated symbolic bound
    res = tvm.arith.detect_iter_map([fld(x, c0), flm(x, c0)], var_dom([(x, c1 * c0)]))

    assert len(res) == 2
    assert_iter_sum_pattern(res[0], c1, 0)
    assert_iter_sum_pattern(res[1], c0, 0)

    res = tvm.arith.detect_iter_map([fld(x * 2, 4), flm(x * 2, 4)], var_dom([(x, 8)]))

    assert len(res) == 2
    assert_iter_sum_pattern(res[0], 4, 0, scale=1)
    assert_iter_sum_pattern(res[1], 2, 0, scale=2)

    res = tvm.arith.detect_iter_map([fld(x * 2, 4) * 4 + flm(x * 2, 4)], var_dom([(x, 8)]))

    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 8, 0, scale=2)

    res = tvm.arith.detect_iter_map([fld(x, flm(flm(y, 8), 6))], var_dom([(x, 24), (y, 8)]))
    assert len(res) == 0


def test_compound():
    x = tvm.tir.Var("x", "int32"), 10
    y = tvm.tir.Var("y", "int32"), 9

    xo, xi = isplit(x, 5)
    yo, yi = isplit(y, 3)
    z = ifuse([yo, xo, yi])

    res = tvm.arith.detect_iter_map([z[0], xi[0]], var_dom([x, y]))

    assert len(res) == 2
    assert_iter_sum_pattern(res[0], 18, 0)
    assert_iter_sum_pattern(res[1], 5, 0)
    # reconstruct the pattern manually
    mx = tvm.arith.IterMark(x[0], 10)
    my = tvm.arith.IterMark(y[0], 9)

    xoscale = 3
    xiscale = 1
    yoscale = 6
    yiscale = 1
    mxo = tvm.arith.IterSplitExpr(mx, 5, 2, xoscale)
    mxi = tvm.arith.IterSplitExpr(mx, 1, 5, xiscale)
    myo = tvm.arith.IterSplitExpr(my, 3, 3, yoscale)
    myi = tvm.arith.IterSplitExpr(my, 1, 3, yiscale)

    mz = tvm.arith.IterMark(tvm.arith.IterSumExpr([myo, mxo, myi], 0), 18)
    sz = tvm.arith.IterSumExpr([tvm.arith.IterSplitExpr(mz, 1, 18, 1)], 0)
    tvm.ir.assert_structural_equal(sz, res[0])


if __name__ == "__main__":
    test_split()
    test_trivial()
    test_fuse()
    test_compound()
