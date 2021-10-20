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
from tvm.tir import floormod, floordiv


def ifuse(inputs, pred_extent=None):
    """Fuse iterators"""
    value, extent = 0, 1
    for i, ext in inputs:
        value = value * ext + i
        extent = extent * ext
    return value, extent if pred_extent is None else pred_extent


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
    assert len(res) == 2
    assert_iter_sum_pattern(res[0], 3, 0)
    assert_iter_sum_pattern(res[1], 1, 3)

    # not independent
    res = tvm.arith.detect_iter_map([x[0], x[0], 3], var_dom([x, y]))
    assert len(res) == 0


def test_fuse():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    c = tvm.tir.SizeVar("c", "int32")
    c0 = tvm.tir.SizeVar("c0", "int32")

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
    c0 = tvm.tir.SizeVar("c0", "int32")
    c1 = tvm.tir.SizeVar("c1", "int32")
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


def test_predicate():
    x = tvm.tir.Var("x", "int32"), 13
    y = tvm.tir.Var("y", "int32"), 10

    res = tvm.arith.detect_iter_map([x[0] * 10 + y[0]], var_dom([x, y]), x[0] * 10 + y[0] < 128)

    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 128, 0)

    # duplicate constraint
    res = tvm.arith.detect_iter_map(
        [x[0] * 10 + y[0]],
        var_dom([x, y]),
        tvm.tir.all(x[0] * 10 + y[0] < 128, x[0] * 10 + y[0] < 64),
    )

    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 64, 0)

    # useless constraint
    res = tvm.arith.detect_iter_map([x[0] * 10 + y[0]], var_dom([x, y]), x[0] * 10 + y[0] < 140)

    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 130, 0)

    i1 = tvm.tir.Var("i1", "int32"), 7
    i2 = tvm.tir.Var("i2", "int32"), 2
    i3 = tvm.tir.Var("i3", "int32"), 4
    i4 = tvm.tir.Var("i4", "int32"), 3
    res = tvm.arith.detect_iter_map(
        [i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0]],
        var_dom([i1, i2, i3, i4]),
        (
            tvm.tir.all(
                i1[0] * 2 + i2[0] < 13,
                i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0] < 128,
                i3[0] * 3 + i4[0] < 10,
            )
        ),
    )
    assert len(res) == 1
    assert_iter_sum_pattern(res[0], 128, 0)

    i1 = tvm.tir.Var("i1", "int32"), 7
    i2 = tvm.tir.Var("i2", "int32"), 2
    i3 = tvm.tir.Var("i3", "int32"), 4
    i4 = tvm.tir.Var("i4", "int32"), 3

    # wrong constraint
    res = tvm.arith.detect_iter_map(
        [i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0]],
        var_dom([i1, i2, i3, i4]),
        (
            tvm.tir.all(
                i1[0] * 2 + i2[0] < 13,
                i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0] < 128,
                i3[0] * 3 + i4[0] < 7,
            )
        ),
    )
    assert len(res) == 0

    # incompatible constraint
    res = tvm.arith.detect_iter_map(
        [i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0]],
        var_dom([i1, i2, i3, i4]),
        (
            tvm.tir.all(
                i1[0] * 2 + i2[0] < 13,
                i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0] < 128,
                i3[0] * 3 + i4[0] < 10,
                i1[0] * 4 + i3[0] < 20,
            )
        ),
    )
    assert len(res) == 0

    res = tvm.arith.detect_iter_map(
        [i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0]],
        var_dom([i1, i2, i3, i4]),
        (
            tvm.tir.all(
                i1[0] * 2 + i2[0] < 13,
                i1[0] * 20 + i2[0] * 10 + i3[0] * 3 + i4[0] < 128,
                i1[0] * 4 + i3[0] < 20,
            )
        ),
    )
    assert len(res) == 0

    # zero iter
    xo = tvm.tir.Var("xo", "int32"), 1
    xi = tvm.tir.Var("xi", "int32"), 129
    y = tvm.tir.Var("y", "int32"), 128

    res = tvm.arith.detect_iter_map(
        [xo[0] * 129 + xi[0], y[0]], var_dom([xo, xi, y]), xo[0] * 129 + xi[0] < 128
    )


def convert_division(divisions):
    if divisions is None or len(divisions) == 0:
        return []
    res = []
    for division in divisions[:-1]:
        res.append(
            [
                tvm.arith.normalize_iter_map_to_expr(division[0].source),
                tvm.arith.normalize_iter_map_to_expr(division[1].source),
            ]
        )
    res.append([divisions[-1][0].extent, divisions[-1][1].extent])
    return res


def create_iter(name, extent):
    return tvm.tir.Var(name, "int32"), extent


def test_subspace_division():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    c = tvm.tir.SizeVar("c", "int32")

    # simple 1.1
    res = tvm.arith.subspace_divide(
        [z * 12 + y * 3 + x + c], var_dom([(x, 3), (y, 4), (z, 5)]), [x]
    )
    res = convert_division(res)
    assert len(res) == 2
    tvm.ir.assert_structural_equal(res[0][0], z * 4 + y)
    tvm.ir.assert_structural_equal(res[0][1], x + c)

    # simple 1.2
    res = tvm.arith.subspace_divide(
        [z * 12 + y * 3 + x + c], var_dom([(x, 3), (y, 4), (z, 5)]), [x], z * 4 + y < 18
    )
    res = convert_division(res)
    assert len(res) == 2
    tvm.ir.assert_structural_equal(res[0][0], z * 4 + y)
    tvm.ir.assert_structural_equal(res[0][1], x + c)
    tvm.ir.assert_structural_equal(res[1][0], z * 4 + y < 18)
    tvm.ir.assert_structural_equal(res[1][1], True)

    # compound 1
    i0 = create_iter("i0", 4)
    j0 = create_iter("j0", 8)
    i3 = create_iter("i3", 2)

    i1, i2 = isplit(j0, 4)
    k0 = ifuse([i0, i1])
    k1 = ifuse([i2, i3])

    # compound 1.1
    res = tvm.arith.subspace_divide([k0[0], k1[0]], var_dom([i0, j0, i3]), [i3[0]])
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], (i0[0] * 2) + floordiv(j0[0], 4))
    tvm.ir.assert_structural_equal(res[0][1], 0)
    tvm.ir.assert_structural_equal(res[1][0], floormod(j0[0], 4))
    tvm.ir.assert_structural_equal(res[1][1], i3[0])

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([i3]))
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0, j0]))
    assert len(res2) == 2

    # compound 1.2
    res = tvm.arith.subspace_divide([k0[0], k1[0]], var_dom([i0, j0, i3]), [j0[0], i3[0]])
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], i0[0])
    tvm.ir.assert_structural_equal(res[0][1], floordiv(j0[0], 4))
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], (floormod(j0[0], 4) * 2) + i3[0])

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([j0, i3]))
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0]))
    assert len(res2) == 2

    # compound 1.3
    res = tvm.arith.subspace_divide([k0[0], k1[0]], var_dom([i0, j0, i3]), [i0[0], i3[0]])
    res = convert_division(res)
    assert len(res) == 0

    # compound 1.4
    res = tvm.arith.subspace_divide([k0[0], k1[0]], var_dom([i0, j0, i3]), [i3[0]], k0[0] < 7)
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], (i0[0] * 2) + floordiv(j0[0], 4))
    tvm.ir.assert_structural_equal(res[0][1], 0)
    tvm.ir.assert_structural_equal(res[1][0], floormod(j0[0], 4))
    tvm.ir.assert_structural_equal(res[1][1], i3[0])
    tvm.ir.assert_structural_equal(res[2][0], (i0[0] * 2) + floordiv(j0[0], 4) < 7)
    tvm.ir.assert_structural_equal(res[2][1], True)

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([i3]))
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0, j0]))
    assert len(res2) == 2

    # compound 1.5
    res = tvm.arith.subspace_divide(
        [k0[0], k1[0]], var_dom([i0, j0, i3]), [j0[0], i3[0]], k1[0] < 7
    )
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], i0[0])
    tvm.ir.assert_structural_equal(res[0][1], floordiv(j0[0], 4))
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], (floormod(j0[0], 4) * 2) + i3[0])
    tvm.ir.assert_structural_equal(res[2][0], True)
    tvm.ir.assert_structural_equal(res[2][1], (floormod(j0[0], 4) * 2) + i3[0] < 7)

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([j0, i3]))
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0]))
    assert len(res2) == 2

    # compound 1.6
    res = tvm.arith.subspace_divide(
        [k0[0], k1[0]], var_dom([i0, j0, i3]), [i3[0]], tvm.tir.all(k0[0] < 7, k1[0] < 7)
    )
    res = convert_division(res)
    assert len(res) == 0

    # compound 2
    j0 = create_iter("j0", 4)
    l0 = create_iter("l0", 2)
    l1 = create_iter("l1", 6)
    j3 = create_iter("j3", 3)

    k0 = ifuse([l0, l1])
    i1, j2 = isplit(k0, 3)
    j1, i1 = isplit(i1, 2)
    i0 = ifuse([j0, j1])
    i2 = ifuse([j2, j3])

    # compound 2.1
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0], i2[0]], var_dom([j0, l0, l1, j3]), [l1[0], j3[0]]
    )
    res = convert_division(res)
    assert len(res) == 4
    tvm.ir.assert_structural_equal(res[0][0], (j0[0] * 2) + l0[0])
    tvm.ir.assert_structural_equal(res[0][1], 0)
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], floordiv(l1[0], 3))
    tvm.ir.assert_structural_equal(res[2][0], 0)
    tvm.ir.assert_structural_equal(res[2][1], (floormod(l1[0], 3) * 3) + j3[0])

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1], res[2][1]], var_dom([l1, j3]))
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0, l0]))
    assert len(res2) == 3

    # compound 2.2
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0], i2[0]], var_dom([j0, l0, l1, j3]), [l0[0], l1[0], j3[0]]
    )
    res = convert_division(res)
    assert len(res) == 4
    tvm.ir.assert_structural_equal(res[0][0], j0[0])
    tvm.ir.assert_structural_equal(res[0][1], floordiv(l0[0] * 6 + l1[0], 6))
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], floormod(floordiv(l0[0] * 6 + l1[0], 3), 2))
    tvm.ir.assert_structural_equal(res[2][0], 0)
    tvm.ir.assert_structural_equal(res[2][1], (floormod(l0[0] * 6 + l1[0], 3) * 3) + j3[0])

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1], res[2][1]], var_dom([l0, l1, j3]))
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0]))
    assert len(res2) == 3

    # compound 2.3
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0], i2[0]], var_dom([j0, l0, l1, j3]), [l0[0], j3[0]]
    )
    res = convert_division(res)
    assert len(res) == 0

    # compound 2.4
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0], i2[0]],
        var_dom([j0, l0, l1, j3]),
        [l1[0], j3[0]],
        tvm.tir.all(i0[0] < 7, i2[0] < 8),
    )
    res = convert_division(res)
    assert len(res) == 4
    tvm.ir.assert_structural_equal(res[0][0], (j0[0] * 2) + l0[0])
    tvm.ir.assert_structural_equal(res[0][1], 0)
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], floordiv(l1[0], 3))
    tvm.ir.assert_structural_equal(res[2][0], 0)
    tvm.ir.assert_structural_equal(res[2][1], (floormod(l1[0], 3) * 3) + j3[0])
    tvm.ir.assert_structural_equal(res[3][0], (j0[0] * 2) + l0[0] < 7)
    tvm.ir.assert_structural_equal(res[3][1], (floormod(l1[0], 3) * 3) + j3[0] < 8)

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1], res[2][1]], var_dom([l1, j3]))
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0, l0]))
    assert len(res2) == 3

    # compound 2.5
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0], i2[0]], var_dom([j0, l0, l1, j3]), [j3[0]], i2[0] < 8
    )
    res = convert_division(res)
    assert len(res) == 0


def test_complex():
    n0 = create_iter("n0", 2)
    n1 = create_iter("n1", 4)

    m0 = ifuse([n0, n1], 6)
    m1 = create_iter("m1", 3)

    l0 = create_iter("l0", 4)
    l1 = create_iter("l1", 8)
    l2 = ifuse([m0, m1], 16)
    l3 = create_iter("l3", 32)

    k0, k4 = isplit(l0, 2)
    k1, k5 = isplit(l1, 2)
    k2, k6 = isplit(l2, 4)
    k3, k7 = isplit(l3, 4)

    j0 = ifuse([k0, k1], 7)
    j1 = ifuse([k2, k3])
    j2 = ifuse([k4, k5])
    j3 = ifuse([k6, k7], 15)

    i0 = ifuse([j0, j1], 200)
    i1 = ifuse([j2, j3], 50)

    res = tvm.arith.detect_iter_map(
        [i0[0], i1[0]],
        var_dom([l0, l1, n0, n1, m1, l3]),
        tvm.tir.all(i0[0] < 200, i1[0] < 50, m0[0] < 6, l2[0] < 16, j0[0] < 7, j3[0] < 15),
    )
    assert len(res) == 2

    n0_mark = tvm.arith.IterMark(n0[0], n0[1])
    n1_mark = tvm.arith.IterMark(n1[0], n1[1])
    l0_mark = tvm.arith.IterMark(l0[0], l0[1])
    l1_mark = tvm.arith.IterMark(l1[0], l1[1])
    m1_mark = tvm.arith.IterMark(m1[0], m1[1])
    l3_mark = tvm.arith.IterMark(l3[0], l3[1])

    m0_expr = tvm.arith.IterSumExpr(
        [
            tvm.arith.IterSplitExpr(n0_mark, 1, n0[1], 4),
            tvm.arith.IterSplitExpr(n1_mark, 1, n1[1], 1),
        ],
        0,
    )
    m0_mark = tvm.arith.IterMark(m0_expr, 6)
    l2_expr = tvm.arith.IterSumExpr(
        [tvm.arith.IterSplitExpr(m0_mark, 1, 6, 3), tvm.arith.IterSplitExpr(m1_mark, 1, m1[1], 1)],
        0,
    )
    l2_mark = tvm.arith.IterMark(l2_expr, 16)
    k0_expr = tvm.arith.IterSplitExpr(l0_mark, 2, 2, 4)
    k1_expr = tvm.arith.IterSplitExpr(l1_mark, 2, 4, 1)
    k2_expr = tvm.arith.IterSplitExpr(l2_mark, 4, 4, 8)
    k3_expr = tvm.arith.IterSplitExpr(l3_mark, 4, 8, 1)
    k4_expr = tvm.arith.IterSplitExpr(l0_mark, 1, 2, 30)
    k5_expr = tvm.arith.IterSplitExpr(l1_mark, 1, 2, 15)
    k6_expr = tvm.arith.IterSplitExpr(l2_mark, 1, 4, 4)
    k7_expr = tvm.arith.IterSplitExpr(l3_mark, 1, 4, 1)

    j0_expr = tvm.arith.IterSumExpr([k0_expr, k1_expr], 0)
    j0_mark = tvm.arith.IterMark(j0_expr, 7)
    i0_expr = tvm.arith.IterSumExpr(
        [tvm.arith.IterSplitExpr(j0_mark, 1, 7, 32), k2_expr, k3_expr], 0
    )

    j3_expr = tvm.arith.IterSumExpr([k6_expr, k7_expr], 0)
    j3_mark = tvm.arith.IterMark(j3_expr, 15)
    i1_expr = tvm.arith.IterSumExpr(
        [k4_expr, k5_expr, tvm.arith.IterSplitExpr(j3_mark, 1, 15, 1)], 0
    )

    i0_mark = tvm.arith.IterMark(i0_expr, i0[1])
    i1_mark = tvm.arith.IterMark(i1_expr, i1[1])

    i0_final = tvm.arith.IterSumExpr([tvm.arith.IterSplitExpr(i0_mark, 1, i0[1], 1)], 0)
    i1_final = tvm.arith.IterSumExpr([tvm.arith.IterSplitExpr(i1_mark, 1, i1[1], 1)], 0)

    tvm.ir.assert_structural_equal(i0_final, res[0])
    tvm.ir.assert_structural_equal(i1_final, res[1])

    # wrong constraint
    res = tvm.arith.detect_iter_map(
        [i0[0], i1[0]],
        var_dom([l0, l1, n0, n1, m1, l3]),
        tvm.tir.all(i0[0] < 200, i1[0] < 50, m0[0] < 9, l2[0] < 16, j0[0] < 7, j3[0] < 14),
    )
    assert len(res) == 0

    # subspace_division
    res = tvm.arith.subspace_divide(
        [i0[0], i1[0]],
        var_dom([l0, l1, n0, n1, m1, l3]),
        [n0[0], n1[0], m1[0], l3[0]],
        tvm.tir.all(m0[0] < 6, l2[0] < 16, j0[0] < 7, j3[0] < 15),
    )
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], floordiv(l0[0], 2) * 4 + floordiv(l1[0], 2))
    tvm.ir.assert_structural_equal(
        res[0][1], (floordiv((n0[0] * 4 + n1[0]) * 3 + m1[0], 4) * 8) + floordiv(l3[0], 4)
    )
    tvm.ir.assert_structural_equal(res[1][0], ((floormod(l0[0], 2) * 2) + floormod(l1[0], 2)))
    tvm.ir.assert_structural_equal(
        res[1][1], ((floormod(((n0[0] * 4 + n1[0]) * 3 + m1[0]), 4) * 4) + floormod(l3[0], 4))
    )
    tvm.ir.assert_structural_equal(res[2][0], (floordiv(l0[0], 2) * 4) + floordiv(l1[0], 2) < 7)
    tvm.ir.assert_structural_equal(
        res[2][1],
        tvm.tir.all(
            n0[0] * 4 + n1[0] < 6,
            (n0[0] * 4 + n1[0]) * 3 + m1[0] < 16,
            floormod(((n0[0] * 4 + n1[0]) * 3 + m1[0]), 4) * 4 + floormod(l3[0], 4) < 15,
        ),
    )

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([n0, n1, m1, l3]), res[2][1])
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([l0, l1]))
    assert len(res2) == 2


def test_normalize_iter_map_to_expr():
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod

    x = tvm.tir.Var("x", "int32"), 10
    y = tvm.tir.Var("y", "int32"), 9

    xo, xi = isplit(x, 5)
    yo, yi = isplit(y, 3)
    z = ifuse([yo, xo, yi])

    res = tvm.arith.detect_iter_map([z[0], xi[0]], var_dom([x, y]))

    tvm.ir.assert_structural_equal(
        tvm.arith.normalize_iter_map_to_expr(res[0]),
        fld(y[0], 3) * 6 + fld(x[0], 5) * 3 + flm(y[0], 3),
    )
    tvm.ir.assert_structural_equal(tvm.arith.normalize_iter_map_to_expr(res[1]), flm(x[0], 5))


def test_inverse_affine_iter_map():
    analyzer = tvm.arith.Analyzer()
    l0 = create_iter("l0", 64)
    l1 = create_iter("l1", 64)
    l2 = create_iter("l2", 64)

    # simple case
    l0_0, l0_1 = isplit(l0, 16)
    l1_0, l1_1 = isplit(l1, 4)
    l0_1_l1_1_fused = ifuse([l0_1, l1_1])

    iter_map = tvm.arith.detect_iter_map([l0_1_l1_1_fused[0], l0_0[0], l1_0[0]], var_dom([l0, l1]))
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 2
    l0_inverse = floormod(floordiv(outputs[0], 4), 16) + outputs[1] * 16
    l1_inverse = floormod(outputs[0], 4) + outputs[2] * 4
    assert analyzer.simplify(res[l0[0]] - l0_inverse) == 0
    assert analyzer.simplify(res[l1[0]] - l1_inverse) == 0

    # compound case
    l0_0, l0_1 = isplit(l0, 16)
    l1_0, l1_1 = isplit(l1, 4)
    l2_1, l2_2 = isplit(l2, 4)
    l2_0, l2_1 = isplit(l2_1, 4)

    l0_1_l2_1_l1_1_l2_0_fused = ifuse([l0_1, l2_1, l1_1, l2_0])

    iter_map = tvm.arith.detect_iter_map(
        [l0_1_l2_1_l1_1_l2_0_fused[0], l0_0[0], l2_2[0], l1_0[0]], var_dom([l0, l1, l2])
    )
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 3
    l0_inverse = floormod(floordiv(outputs[0], 64), 16) + outputs[1] * 16
    l1_inverse = floormod(floordiv(outputs[0], 4), 4) + outputs[3] * 4
    l2_inverse = (
        floormod(outputs[0], 4) * 16 + floormod(floordiv(outputs[0], 16), 4) * 4 + outputs[2]
    )

    assert analyzer.simplify(res[l0[0]] - l0_inverse) == 0
    assert analyzer.simplify(res[l1[0]] - l1_inverse) == 0
    assert analyzer.simplify(res[l2[0]] - l2_inverse) == 0

    # diamond-shape DAG
    l0_0, l0_1 = isplit(l0, 16)
    l1 = ifuse([l0_1, l0_0])
    l1_0, l1_1 = isplit(l1, 8)
    l2 = ifuse([l1_1, l1_0])

    iter_map = tvm.arith.detect_iter_map([l2[0]], var_dom([l0]))
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 1
    l1_inverse = floormod(outputs[0], 8) * 8 + floormod(floordiv(outputs[0], 8), 8)
    l0_inverse = floormod(l1_inverse, 4) * 16 + floormod(floordiv(l1_inverse, 4), 16)

    assert analyzer.simplify(res[l0[0]] - l0_inverse) == 0


if __name__ == "__main__":
    test_split()
    test_trivial()
    test_fuse()
    test_compound()
    test_predicate()
    test_normalize_iter_map_to_expr()
    test_subspace_division()
    test_complex()
    test_inverse_affine_iter_map()
