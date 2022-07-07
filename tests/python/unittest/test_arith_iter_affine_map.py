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
from xml import dom
import tvm
import tvm.testing
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


def convert_iter_expr(expr):
    return tvm.arith.normalize_iter_map_to_expr(expr)


def assert_iter_sum_pattern(
    expect_dict, dom_map, predicate=True, check_level="surjective", simplify_trivial_iterators=True
):
    keys = list(expect_dict.keys())
    res = tvm.arith.detect_iter_map(
        keys,
        dom_map,
        predicate=predicate,
        check_level=check_level,
        simplify_trivial_iterators=simplify_trivial_iterators,
    )
    indices = res.indices
    assert len(indices) == len(keys), res.errors
    print(indices)
    for i, input_iter in enumerate(keys):
        spec = expect_dict[input_iter]
        (
            extent,
            base,
        ) = spec[0:2]
        scale = spec[2] if len(spec) > 2 else 1
        expect_iter = spec[3] if len(spec) > 3 else None
        sum_expr = indices[i]
        assert isinstance(sum_expr, tvm.arith.IterSumExpr)
        if extent == 1:
            assert len(sum_expr.args) == 0
        else:
            assert len(sum_expr.args) == 1
            tvm.testing.assert_prim_expr_equal(sum_expr.args[0].extent, extent)
            tvm.testing.assert_prim_expr_equal(sum_expr.args[0].scale, scale)
        tvm.testing.assert_prim_expr_equal(sum_expr.base, base)
        if expect_iter is not None:
            if not isinstance(expect_iter, tvm.arith.IterMapExpr):
                sum_expr = convert_iter_expr(sum_expr)
            tvm.ir.assert_structural_equal(sum_expr, expect_iter)


def assert_iter_sum_failure(iters, dom_map, predicate=True, check_level="surjective"):
    res = tvm.arith.detect_iter_map(
        list(iters), dom_map, predicate=predicate, check_level=check_level
    ).indices
    assert len(res) == 0


def test_trivial():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")
    dom_map = var_dom([(x, 3), (y, 4), (z, 1)])

    assert_iter_sum_pattern({x: (3, 0), y: (4, 0), 3: (1, 3)}, dom_map)
    assert_iter_sum_pattern({x: (3, 0), 3: (1, 3)}, dom_map)

    # not independent
    assert_iter_sum_failure([x, x, 3], dom_map)

    assert_iter_sum_pattern(
        {x: (3, 0), y: (4, 0)}, dom_map, check_level="bijective", simplify_trivial_iterators=True
    )
    assert_iter_sum_pattern(
        {x: (3, 0), y: (4, 0)}, dom_map, check_level="bijective", simplify_trivial_iterators=False
    )
    assert_iter_sum_failure([x, z], dom_map, check_level="bijective")


def test_fuse():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    c = tvm.tir.SizeVar("c", "int32")
    c0 = tvm.tir.SizeVar("c0", "int32")

    assert_iter_sum_pattern({y * 3 + 1 + c + x: (12, 1 + c)}, var_dom([(x, 3), (y, 4)]))

    assert_iter_sum_pattern({ifuse([(x, 3), (y, 4)])[0]: (12, 0)}, var_dom([(x, 3), (y, 4)]))

    # fuse with symbolic factor
    assert_iter_sum_pattern({(y + 1) * c + x: (4 * c, c)}, var_dom([(x, c), (y, 4)]))

    # duplication
    assert_iter_sum_failure([y * 3 + x, y], var_dom([(x, 3), (y, 4)]))
    assert_iter_sum_failure([y, x + 1, y], var_dom([(x, 3), (y, 4)]))

    # factor mismatch
    assert_iter_sum_failure([y * 4 + x], var_dom([(x, 3), (y, 4)]))

    # simple stride pattern
    assert_iter_sum_pattern({x * 4 + y * 2: (6, 0, 2, (x * 2 + y) * 2)}, var_dom([(x, 3), (y, 2)]))

    # simple stride pattern with symbolic
    assert_iter_sum_pattern(
        {x * 2 * c0 + y * 2: (3 * c0, 0, 2, (x * c0 + y) * 2)}, var_dom([(x, 3), (y, c0)])
    )


def test_split():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    c0 = tvm.tir.SizeVar("c0", "int32")
    c1 = tvm.tir.SizeVar("c1", "int32")
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod

    assert_iter_sum_pattern({fld(x, 3): (8, 0), flm(x, 3) * 2 + c1: (3, c1, 2)}, var_dom([(x, 24)]))

    assert_iter_sum_pattern(
        {fld(x, 6): (4, 0), fld(flm(x, 6), 2): (3, 0), flm(x, 2): (2, 0)}, var_dom([(x, 24)])
    )

    # simple symbolic bound
    # TODO(tvm-team) improve symbolic divisible check to enable
    # more complicated symbolic bound
    assert_iter_sum_pattern({fld(x, c0): (c1, 0), flm(x, c0): (c0, 0)}, var_dom([(x, c1 * c0)]))

    assert_iter_sum_pattern({fld(x * 2, 4): (4, 0, 1), flm(x * 2, 4): (2, 0, 2)}, var_dom([(x, 8)]))

    assert_iter_sum_pattern(
        {
            fld(x * 2, 4) * 4 + flm(x * 2, 4): (8, 0, 2),
        },
        var_dom([(x, 8)]),
    )

    assert_iter_sum_failure([fld(x, flm(flm(y, 8), 6))], var_dom([(x, 24), (y, 8)]))

    # domain of x is undefined
    assert_iter_sum_pattern(
        {fld(flm(x, 49) + y, 49): (1, fld(flm(x, 49) + y, 49))}, var_dom([(y, 1)])
    )


def test_compound():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")

    xo, xi = isplit((x, 10), 5)
    yo, yi = isplit((y, 9), 3)
    z = ifuse([yo, xo, yi])

    # reconstruct the pattern manually
    mx = tvm.arith.IterMark(x, 10)
    my = tvm.arith.IterMark(y, 9)
    xoscale = 3
    yoscale = 6
    yiscale = 1
    mxo = tvm.arith.IterSplitExpr(mx, 5, 2, xoscale)
    myo = tvm.arith.IterSplitExpr(my, 3, 3, yoscale)
    myi = tvm.arith.IterSplitExpr(my, 1, 3, yiscale)
    mz = tvm.arith.IterMark(tvm.arith.IterSumExpr([myo, mxo, myi], 0), 18)
    sz = tvm.arith.IterSumExpr([tvm.arith.IterSplitExpr(mz, 1, 18, 1)], 0)
    assert_iter_sum_pattern({z[0]: (18, 0, 1, sz), xi[0]: (5, 0)}, var_dom([(x, 10), (y, 9)]))


def test_predicate():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")

    # available contraints
    # upper bound only
    assert_iter_sum_pattern(
        {x * 10 + y: (128, 0)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 + y < 128
    )

    assert_iter_sum_pattern(
        {x * 10 + y: (128, 0)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 + y <= 127
    )

    # lower bound only
    assert_iter_sum_pattern(
        {x * 10 + y: (124, 6)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 + y > 5
    )

    assert_iter_sum_pattern(
        {x * 10 + y: (124, 6)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 + y >= 6
    )

    # lower bound + upper bound
    assert_iter_sum_pattern(
        {x * 10 + y: (122, 6)},
        var_dom([(x, 13), (y, 10)]),
        predicate=tvm.tir.And(x * 10 + y > 5, x * 10 + y < 128),
    )

    assert_iter_sum_pattern(
        {x * 10 + y: (122, 6)},
        var_dom([(x, 13), (y, 10)]),
        predicate=tvm.tir.And(x * 10 + y >= 6, x * 10 + y <= 127),
    )

    # constraint on one fused iter
    i = tvm.tir.Var("i", "int32")
    j = tvm.tir.Var("j", "int32")
    k = tvm.tir.Var("k", "int32")
    assert_iter_sum_pattern(
        {i * 8 + j * 2 + k: (88, 1)},
        var_dom([(i, 11), (j, 5), (k, 2)]),
        predicate=tvm.tir.all(1 <= j * 2 + k, j * 2 + k < 9),
    )

    # constraint on single var
    assert_iter_sum_pattern({i: (10, 0)}, var_dom([(i, 48)]), predicate=i < 10)

    # iterations are subparts of constraint, invalid case 1
    assert_iter_sum_failure(
        [i, j, k],
        var_dom([(i, 128), (j, 128), (k, 128)]),
        predicate=tvm.tir.all(i * 16384 + j * 128 + k < 100),
    )

    # iterations are subparts of constraint, invalid case 2
    assert_iter_sum_failure(
        [i * 128 + j, k],
        var_dom([(i, 128), (j, 128), (k, 128)]),
        predicate=i * 16384 + j * 128 + k < 100,
    )

    # irrelavant predicate
    assert_iter_sum_pattern({i + j: (1, j)}, var_dom([(i, 1)]), predicate=j <= 24)

    # constraint on nested fused iters
    assert_iter_sum_pattern(
        {i * 8 + j * 2 + k: (22, 3)},
        var_dom([(i, 11), (j, 5), (k, 2)]),
        predicate=tvm.tir.all(
            1 <= j * 2 + k, j * 2 + k < 9, 3 <= i * 8 + j * 2 + k, i * 8 + j * 2 + k < 25
        ),
    )

    # duplicate constraint on one fused iter
    assert_iter_sum_pattern(
        {i * 6 + j * 2 + k: (66, 2)},
        var_dom([(i, 11), (j, 5), (k, 2)]),
        predicate=tvm.tir.all(1 <= j * 2 + k, 2 <= j * 2 + k, j * 2 + k < 8, j * 2 + k < 9),
    )

    # duplicate constraint on nested fused iters
    assert_iter_sum_pattern(
        {i * 6 + j * 2 + k: (15, 3)},
        var_dom([(i, 11), (j, 5), (k, 2)]),
        predicate=tvm.tir.all(
            1 <= j * 2 + k,
            2 <= j * 2 + k,
            j * 2 + k < 8,
            j * 2 + k < 9,
            3 <= i * 6 + j * 2 + k,
            i * 6 + j * 2 + k < 25,
            1 <= i * 6 + j * 2 + k,
            i * 6 + j * 2 + k < 18,
        ),
    )

    # constraint on non-disjoint fused iters should fail
    assert_iter_sum_failure(
        [i * 8 + j * 2 + k],
        var_dom([(i, 11), (j, 5), (k, 2)]),
        predicate=tvm.tir.all(2 <= j * 2 + k, 0 <= i * 4 + j),
    )

    # constraint on many disjoint fused iters, case 1
    # i4 * 6 + i5 in [3, 9), extent=6 (= scale of i2)
    # i2 * 30 + i3 * 15 in [30, 90), extent=60 (= scale of i1)
    # i1 * 60 in [60, 240), extent=180 (= scale of i0)
    i0 = tvm.tir.Var("i0", "int32")
    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    i3 = tvm.tir.Var("i3", "int32")
    i4 = tvm.tir.Var("i4", "int32")
    i5 = tvm.tir.Var("i5", "int32")
    assert_iter_sum_pattern(
        {i0 * 180 + i1 * 60 + i2 * 30 + i3 * 15 + i4 * 6 + i5: (540, 93)},
        var_dom([(i0, 3), (i1, 4), (i2, 3), (i3, 2), (i4, 3), (i5, 6)]),
        predicate=tvm.tir.all(1 <= i1, 2 <= i2 * 2 + i3, 3 <= i4 * 6 + i5),
    )

    # constraint on many disjoint fused iters, case 2
    assert_iter_sum_pattern(
        {i0 * 45 + i1 * 45 + i2 * 9 + i3 * 4 + i4: (135, 28)},
        var_dom([(i0, 3), (i1, 2), (i2, 5), (i3, 3), (i4, 4)]),
        predicate=tvm.tir.all(
            3 <= i1 * 5 + i2, i1 * 5 + i2 < 8, 1 <= i3 * 4 + i4, i3 * 4 + i4 < 10
        ),
    )

    # constraint on split iters
    assert_iter_sum_pattern(
        {i % 16: (7, 3), i // 16: (8, 4)},
        var_dom([(i, 1024)]),
        predicate=tvm.tir.all(3 <= i % 16, i % 16 < 10, 4 <= i // 16, i // 16 < 12),
        check_level="bijective",
    )

    # constraint on split iters, nested case 1
    assert_iter_sum_pattern(
        {(i * 32 + j) % 16: (7, 3)},
        var_dom([(i, 5), (j, 32)]),
        predicate=tvm.tir.all(3 <= (i * 32 + j) % 16, (i * 32 + j) % 16 < 10),
    )

    # constraint on split iters, nested case 2
    assert_iter_sum_failure(
        [
            (i * 32 + j) % 16,
        ],
        var_dom([(i, 5), (j, 32)]),
        predicate=tvm.tir.all(1 <= i * 32 + j, i * 32 + j <= 32),
        check_level="bijective",
    )
    assert_iter_sum_pattern(
        {(i * 32 + j) % 16: (16, 0)},
        var_dom([(i, 5), (j, 32)]),
        predicate=tvm.tir.all(1 <= i * 32 + j, i * 32 + j <= 32),
    )
    assert_iter_sum_pattern(
        {(i * 32 + j - 1) % 16: (16, 0), (i * 32 + j - 1) // 16: (4, 0)},
        var_dom([(i, 5), (j, 32)]),
        predicate=tvm.tir.all(1 <= i * 32 + j, i * 32 + j <= 64),
    )

    # non-standard form of predicate
    assert_iter_sum_pattern(
        {x * 10 + y: (128, 0)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 < 128 - y
    )

    # duplicate constraint
    assert_iter_sum_pattern(
        {x * 10 + y: (64, 0)},
        var_dom([(x, 13), (y, 10)]),
        predicate=tvm.tir.all(x * 10 + y < 128, x * 10 + y < 64),
    )

    # useless constraint
    assert_iter_sum_pattern(
        {x * 10 + y: (130, 0)}, var_dom([(x, 13), (y, 10)]), predicate=x * 10 + y < 140
    )

    i1 = tvm.tir.Var("i1", "int32")
    i2 = tvm.tir.Var("i2", "int32")
    i3 = tvm.tir.Var("i3", "int32")
    i4 = tvm.tir.Var("i4", "int32")
    assert_iter_sum_pattern(
        {i1 * 20 + i2 * 10 + i3 * 3 + i4: (128, 0)},
        var_dom([(i1, 7), (i2, 2), (i3, 4), (i4, 3)]),
        predicate=(
            tvm.tir.all(
                i1 * 2 + i2 < 13,
                i1 * 20 + i2 * 10 + i3 * 3 + i4 < 128,
                i3 * 3 + i4 < 10,
            )
        ),
    )

    # wrong constraint
    assert_iter_sum_failure(
        [i1 * 20 + i2 * 10 + i3 * 3 + i4],
        var_dom([(i1, 7), (i2, 2), (i3, 4), (i4, 3)]),
        predicate=(
            tvm.tir.all(
                i1 * 2 + i2 < 13,
                i1 * 20 + i2 * 10 + i3 * 3 + i4 < 128,
                i3 * 3 + i4 < 7,
            )
        ),
    )

    # incompatible constraint
    assert_iter_sum_failure(
        [i1 * 20 + i2 * 10 + i3 * 3 + i4],
        var_dom([(i1, 7), (i2, 2), (i3, 4), (i4, 3)]),
        predicate=(
            tvm.tir.all(
                i1 * 2 + i2 < 13,
                i1 * 20 + i2 * 10 + i3 * 3 + i4 < 128,
                i3 * 3 + i4 < 10,
                i1 * 4 + i3 < 20,
            )
        ),
    )
    assert_iter_sum_failure(
        [i1 * 20 + i2 * 10 + i3 * 3 + i4],
        var_dom([(i1, 7), (i2, 2), (i3, 4), (i4, 3)]),
        predicate=(
            tvm.tir.all(
                i1 * 2 + i2 < 13,
                i1 * 20 + i2 * 10 + i3 * 3 + i4 < 128,
                i1 * 4 + i3 < 20,
            )
        ),
    )

    # zero iter
    xo = tvm.tir.Var("xo", "int32")
    xi = tvm.tir.Var("xi", "int32")
    y = tvm.tir.Var("y", "int32")
    assert_iter_sum_pattern(
        {xo * 129 + xi: (128, 0), y: (128, 0)},
        var_dom([(xo, 1), (xi, 129), (y, 128)]),
        predicate=xo * 129 + xi < 128,
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

    assert_iter_sum_pattern
    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([i3])).indices
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0, j0])).indices
    assert len(res2) == 2

    # compound 1.2
    res = tvm.arith.subspace_divide([k0[0], k1[0]], var_dom([i0, j0, i3]), [j0[0], i3[0]])
    res = convert_division(res)
    assert len(res) == 3
    tvm.ir.assert_structural_equal(res[0][0], i0[0])
    tvm.ir.assert_structural_equal(res[0][1], floordiv(j0[0], 4))
    tvm.ir.assert_structural_equal(res[1][0], 0)
    tvm.ir.assert_structural_equal(res[1][1], (floormod(j0[0], 4) * 2) + i3[0])

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([j0, i3])).indices
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0])).indices
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

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([i3])).indices
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0, j0])).indices
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

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1]], var_dom([j0, i3])).indices
    assert len(res1) == 2
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0]], var_dom([i0])).indices
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

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1], res[2][1]], var_dom([l1, j3])).indices
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0, l0])).indices
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
    tvm.ir.assert_structural_equal(res[1][1], floordiv(floormod(l0[0] * 6 + l1[0], 6), 3))
    tvm.ir.assert_structural_equal(res[2][0], 0)
    tvm.ir.assert_structural_equal(res[2][1], (floormod(l0[0] * 6 + l1[0], 3) * 3) + j3[0])

    res1 = tvm.arith.detect_iter_map(
        [res[0][1], res[1][1], res[2][1]], var_dom([l0, l1, j3])
    ).indices
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0])).indices
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

    res1 = tvm.arith.detect_iter_map([res[0][1], res[1][1], res[2][1]], var_dom([l1, j3])).indices
    assert len(res1) == 3
    res2 = tvm.arith.detect_iter_map([res[0][0], res[1][0], res[2][0]], var_dom([j0, l0])).indices
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

    assert_iter_sum_pattern(
        {i0[0]: (200, 0, 1, i0_final), i1[0]: (50, 0, 1, i1_final)},
        var_dom([l0, l1, n0, n1, m1, l3]),
        predicate=tvm.tir.all(
            i0[0] < 200, i1[0] < 50, m0[0] < 6, l2[0] < 16, j0[0] < 7, j3[0] < 15
        ),
    )

    # wrong constraint
    assert_iter_sum_failure(
        [i0[0], i1[0]],
        var_dom([l0, l1, n0, n1, m1, l3]),
        tvm.tir.all(i0[0] < 200, i1[0] < 50, m0[0] < 9, l2[0] < 16, j0[0] < 7, j3[0] < 14),
    )

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

    assert_iter_sum_pattern(
        {res[0][1]: (32, 0), res[1][1]: (15, 0)}, var_dom([n0, n1, m1, l3]), res[2][1]
    )
    assert_iter_sum_pattern({res[0][0]: (8, 0), res[1][0]: (4, 0)}, var_dom([l0, l1]))


def test_normalize_iter_map_to_expr():
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod

    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")

    xo, xi = isplit((x, 10), 5)
    yo, yi = isplit((y, 9), 3)
    z = ifuse([yo, xo, yi])
    res = tvm.arith.detect_iter_map([z[0], xi[0]], var_dom([(x, 10), (y, 9)]))

    tvm.ir.assert_structural_equal(
        tvm.arith.normalize_iter_map_to_expr(res.indices[0]),
        fld(y, 3) * 6 + fld(x, 5) * 3 + flm(y, 3),
    )
    tvm.ir.assert_structural_equal(tvm.arith.normalize_iter_map_to_expr(res.indices[1]), flm(x, 5))

    # iter mark wrap a complex expr
    split = tvm.arith.IterSplitExpr(tvm.arith.IterMark(x * y + 1, 1024), 1, 1024, 1)
    tvm.ir.assert_structural_equal(tvm.arith.normalize_iter_map_to_expr(split), x * y + 1)


def test_inverse_affine_iter_map():
    analyzer = tvm.arith.Analyzer()
    l0 = create_iter("l0", 64)
    l1 = create_iter("l1", 64)
    l2 = create_iter("l2", 64)

    # simple case
    l0_0, l0_1 = isplit(l0, 16)
    l1_0, l1_1 = isplit(l1, 4)
    l0_1_l1_1_fused = ifuse([l0_1, l1_1])

    iter_map = tvm.arith.detect_iter_map(
        [l0_1_l1_1_fused[0], l0_0[0], l1_0[0]], var_dom([l0, l1])
    ).indices
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 2
    l0_inverse = floordiv(outputs[0], 4) + outputs[1] * 16
    l1_inverse = floormod(outputs[0], 4) + outputs[2] * 4
    assert analyzer.can_prove_equal(res[l0[0]], l0_inverse)
    assert analyzer.can_prove_equal(res[l1[0]], l1_inverse)

    # compound case
    l0_0, l0_1 = isplit(l0, 16)
    l1_0, l1_1 = isplit(l1, 4)
    l2_1, l2_2 = isplit(l2, 4)
    l2_0, l2_1 = isplit(l2_1, 4)

    l0_1_l2_1_l1_1_l2_0_fused = ifuse([l0_1, l2_1, l1_1, l2_0])

    iter_map = tvm.arith.detect_iter_map(
        [l0_1_l2_1_l1_1_l2_0_fused[0], l0_0[0], l2_2[0], l1_0[0]], var_dom([l0, l1, l2])
    ).indices
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 3
    l0_inverse = floordiv(outputs[0], 64) + outputs[1] * 16
    l1_inverse = floormod(floordiv(outputs[0], 4), 4) + outputs[3] * 4
    l2_inverse = (
        floormod(outputs[0], 4) * 16 + floormod(floordiv(outputs[0], 16), 4) * 4 + outputs[2]
    )

    assert analyzer.can_prove_equal(res[l0[0]], l0_inverse)
    assert analyzer.can_prove_equal(res[l1[0]], l1_inverse)
    assert analyzer.can_prove_equal(res[l2[0]], l2_inverse)

    # diamond-shape DAG
    l0_0, l0_1 = isplit(l0, 16)
    l1 = ifuse([l0_1, l0_0])
    l1_0, l1_1 = isplit(l1, 8)
    l2 = ifuse([l1_1, l1_0])

    iter_map = tvm.arith.detect_iter_map([l2[0]], var_dom([l0])).indices
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    assert len(res) == 1
    l1_inverse = floormod(outputs[0], 8) * 8 + floordiv(outputs[0], 8)
    l0_inverse = floormod(l1_inverse, 4) * 16 + floordiv(l1_inverse, 4)

    assert analyzer.can_prove_equal(res[l0[0]], l0_inverse)


def test_inverse_affine_map_trivial_iter():
    analyzer = tvm.arith.Analyzer()
    l0 = create_iter("l0", 64)
    l1 = create_iter("l1", 64)
    iter_map = tvm.arith.detect_iter_map([0, l0[0], l1[0]], var_dom([l0, l1])).indices
    outputs = [tvm.tir.Var("output_{}".format(i), "int32") for i in range(len(iter_map))]
    res = tvm.arith.inverse_affine_iter_map(iter_map, outputs)
    # output_0 is expected to be constant and it is not included in the inverse map
    assert len(res) == 2
    assert analyzer.can_prove_equal(res[l0[0]], outputs[1])
    assert analyzer.can_prove_equal(res[l1[0]], outputs[2])


def test_free_variables():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    z = tvm.tir.Var("z", "int32")

    # illegal iter if z is within dom
    assert_iter_sum_failure([z * 19 + y * 3 + x], var_dom([(x, 3), (y, 3), (z, 3)]))

    # iter is valid if z is free, even there are linear forms of z
    assert_iter_sum_pattern(
        {z * 19 + y * 3 + x: (9, z * 19)},
        var_dom(
            [
                (x, 3),
                (y, 3),
            ]
        ),
    )
    assert_iter_sum_pattern(
        {z * z + y * 3 + x: (9, z * z)},
        var_dom(
            [
                (x, 3),
                (y, 3),
            ]
        ),
    )


def test_padding():
    x = tvm.tir.Var("x", "int32")
    y = tvm.tir.Var("y", "int32")
    fld = tvm.tir.floordiv
    flm = tvm.tir.floormod

    # left padding only, offset divisible
    sum = 64 + y
    dom_map = var_dom([(y, 192)])
    assert_iter_sum_pattern(
        {fld(sum, 32): (6, 2, 1), flm(sum, 32): (32, 0, 1)},
        dom_map,
        check_level="bijective",
    )

    # left padding only, offset non-divisible
    sum = 80 + y
    dom_map = var_dom([(y, 176)])
    assert_iter_sum_pattern(
        {fld(sum, 32): (6, 2, 1)},
        dom_map,
    )
    assert_iter_sum_pattern(
        {flm(fld(sum, 2), 16): (16, 0, 1), flm(sum, 2): (2, 0, 1)},
        dom_map,
    )
    assert_iter_sum_failure({fld(sum, 32), flm(sum, 32)}, dom_map)
    assert_iter_sum_failure({fld(sum, 32), fld(sum, 4)}, dom_map)

    # right padding only, offset divisible
    sum = x * 32 + y * 8
    dom_map = var_dom([(x, 5), (y, 4)])
    assert_iter_sum_pattern(
        {fld(sum, 16): (10, 0, 1), flm(sum, 16): (2, 0, 8)},
        dom_map,
    )
    assert_iter_sum_failure({fld(sum, 5)}, dom_map)

    # right padding only, offset non-divisible
    dom_map = var_dom([(x, 26)])
    assert_iter_sum_pattern(
        {fld(x, 15): (2, 0, 1)},
        dom_map,
    )
    assert_iter_sum_pattern(
        {flm(fld(x, 3), 5): (5, 0, 1), flm(x, 3): (3, 0, 1)},
        dom_map,
    )

    # padding constants on both side
    sum = x + 71
    dom_map = var_dom([(x, 45)])
    assert_iter_sum_pattern({fld(sum, 32): (2, 2, 1)}, dom_map)
    assert_iter_sum_pattern(
        {flm(fld(x, 4), 8): (8, 0, 1), flm(x, 4): (4, 0, 1)},
        dom_map,
    )

    # padding for free iteration part
    sum = x * 360 + y
    dom_map = var_dom([(y, 360)])
    assert_iter_sum_pattern({fld(sum, 16): (23, fld(x * 360 - flm(x, 2) * 8, 16), 1)}, dom_map)
    assert_iter_sum_pattern({flm(x * 360 + y, 16): (16, 0, 1)}, dom_map)

    # multiple split with same mark offset, could
    # be surjective on missing (padded // LCM)
    assert_iter_sum_pattern(
        {
            flm(x + 10, 3): (3, 0),
            flm(fld(x + 10, 3), 4): (4, 0),
            flm(fld(fld(x + 10, 3), 4), 5): (5, 0),
        },
        var_dom([(x, 240)]),
    )
    assert_iter_sum_failure(
        {
            flm(x + 10, 3),
            flm(fld(x + 10, 3), 4),
            flm(fld(fld(x + 10, 3), 4), 5),
            fld(fld(fld(x + 10, 3), 4), 5),
        },
        var_dom([(x, 240)]),
    )

    # different offsets on splits
    assert_iter_sum_pattern(
        {
            flm(x + 1, 3): (3, 0),
            flm(fld(x + 10, 3) + 2, 4): (4, 0),
            flm(fld(fld(x + 10, 3), 4) + 3, 5): (5, 0),
        },
        var_dom([(x, 240)]),
    )

    # original extent is smaller than the divident
    # it is not surjective wrt to the region [0, 16)
    assert_iter_sum_failure({flm(x, 16)}, var_dom([(x, 3)]))


if __name__ == "__main__":
    tvm.testing.main()
