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


class CanonicalChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, expected):
        res = self.analyzer.canonical_simplify(data)
        expected = tvm.runtime.convert(expected)
        assert tvm.ir.structural_equal(res, expected), "\ndata={}\nres={}\nexpected={}".format(
            data, res, expected
        )


def test_mul_sum_simplify():
    ck = CanonicalChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    ck.verify(2 + (3 * x + z + y + 1) * 4 + x, x * 13 + z * 4 + y * 4 + 6)
    ck.verify(x * 3 - 4 * x + 1, 1 - x)
    ck.verify(y + x * 3 - 5 * x + 1 + y, y * 2 + 1 - x * 2)
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # trucdiv
    ck.verify(tdiv(x + y + x + y * 3, 2), y * 2 + x)
    ck.verify(tmod(x + y + x + y * 3, 2), 0)

    # floordiv
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    ck.verify(flm(x + x + y * 3, 2), flm(y * 3, 2))
    ck.verify(fld(x + y + x + y * 3, 2), y * 2 + x)
    ck.verify(flm(x + y + x + y * 3, 2), 0)
    ck.verify(fld(x + x + y * 3, 2), fld(y * 3, 2) + x)


def test_split_index_simplify():
    ck = CanonicalChecker()
    x, y, z = te.var("x"), te.var("y"), te.var("z")

    # trucdiv
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod

    # split div const
    ck.verify(tdiv(x, 3) * 3 + tmod(x, 3), x)
    ck.verify(tdiv(x, 6) * 6 + tmod(tdiv(x, 3), 2) * 3 + tmod(x, 3), x)
    ck.verify(tdiv(tdiv(tmod(x, 16), 2) * 2, 4), tdiv(tmod(x, 16), 4))
    ck.verify(tdiv(tmod(x, 2), 8), 0)
    ck.verify(tdiv(tmod(x, 2), 7), 0)
    ck.verify(tdiv(tdiv(tmod(x, 16), 2) * 2, 6), tdiv(tmod(x, 16), 6))

    # split mod const
    ck.verify(tmod((x * 8), 16), tmod(x, 2) * 8)
    ck.verify(tmod(x * 8, 2), 0)

    # simplify then fold
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000))
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000))
    ck.verify(tdiv(x * 4 + y, 2) * 2 + tmod(x * 4 + y, 2), x * 4 + y)
    # complex fold
    ck.verify(tdiv(z * 9 + y, 2) * 2 + tmod(z * 9 + y, 2), z * 9 + y)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(-100, 1000), True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(-100, 1000), True)
    ck.verify(tdiv(x * 4 + y, 2) * 2 + tmod(x * 4 + y, 2), x * 4 + y)

    # floordiv
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    ck.verify(fld(x * 5, 2), fld(x * 5, 2))
    ck.verify(fld(x, 3) * 3 + flm(x, 3), x)
    ck.verify(fld(x, 6) * 6 + flm(fld(x, 3), 2) * 3 + flm(x, 3), x)
    ck.verify(fld(fld(flm(x, 16), 2) * 2, 4), fld(flm(x, 16), 4))
    ck.verify(fld(flm(x, 2), 8), 0)
    ck.verify(fld(flm(x, 2), 7), 0)
    ck.verify(fld(fld(flm(x, 16), 2) * 2, 6), fld(flm(x, 16), 6))

    # cannot simplify mixed case, unless we canonicalize into one mode.
    ck.verify(tdiv(x, 6) * 2 + tmod(fld(x, 3), 2), tdiv(x, 6) * 2 + tmod(fld(x, 3), 2))

    ck.verify(tmod(-x, 2), tmod(x, -2) * -1)


def test_div_simplify():
    ck = CanonicalChecker()
    x = te.var("x")
    tdiv = tvm.tir.truncdiv

    # truc div
    ck.verify(tdiv(16 + 48 * x, 16), x * 3 + 1)
    # (17+48*x)/16 is not simplifiable for arbitrary x because when 17+48*x<0
    # (17+48*x)/16 != 1+3*x
    ck.verify(tdiv(17 + 48 * x, 16), tdiv(x * 48 + 17, 16))
    # However, when x >= 0, then 17+48*x >= 0 and (17+48*x)/16 can be simplified
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 10))
    ck.verify(tdiv(17 + 48 * x, 16), x * 3 + 1)
    # Trying expressions that are not simplifiable for any values of the variables
    ck.verify(tdiv(17 + 47 * x, 16), tdiv(x * 47 + 17, 16))

    # floordiv
    fld = tvm.te.floordiv
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 10000), True)
    ck.verify(fld(16 + 48 * x, 16), x * 3 + 1)
    ck.verify(fld(17 + 48 * x, 16), x * 3 + 1)
    ck.verify(fld(17 + 47 * x, 16), fld(x * 47 + 17, 16))


def test_floormod_simplify():
    ck = CanonicalChecker()
    flm = tvm.te.floormod
    x, y = te.var("x"), te.var("y")
    ck.verify(flm(flm((x * 4) + y - 466036, 24528) - 24512, 16), flm((x * 4) + y + 12, 16))
    ck.verify(flm(flm((x * 4), 16), 8), flm(x, 2) * 4)

    ck.verify(flm(-x, 2), flm(x, -2) * -1)


def test_canonical_mixed():
    ck = CanonicalChecker()
    x = te.var("x")
    z = tvm.tir.const(3, "int32")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    ck.verify(tdiv(x, (z * z)) - tdiv(x, (z * z)), 0)
    ck.verify(tdiv(x, (z + z)) - tdiv(x, (z + z)), 0)
    ck.verify(x - 2 < 3, x < 5)
    ck.verify(tvm.te.max(x, 1) - tvm.te.max(x, 1), 0)
    ck.verify(tvm.te.min(x, 1) - tvm.te.min(x, 1), 0)
    ck.verify(x * x - x * x, 0)
    ck.verify(tmod(tdiv(tmod(x, 20), 2) * 2, 4), tdiv(tmod(x, 4), 2) * 2)

    fld = tvm.te.floordiv
    ck.verify(fld(x, (z * z)) - fld(x, (z * z)), 0)
    ck.verify(fld(x, (z + z)) - fld(x, (z + z)), 0)


def test_reduce_combiner_simplify():
    ck = CanonicalChecker()
    dummy = te.var("dummy")
    comm_reducer = te.comm_reducer
    prod = comm_reducer(lambda x, y: x * y, lambda t0: tvm.tir.const(1, t0))

    sum_or_prod = comm_reducer(
        lambda x, y: tvm.tir.Select(dummy < 0, x + y, x * y),
        lambda t0: tvm.tir.Select(dummy < 0, tvm.tir.const(0, t0), tvm.tir.const(1, t0)),
    )
    sum_and_prod = comm_reducer(
        lambda x, y: (x[0] + y[0], x[1] * y[1]),
        lambda t0, t1: (tvm.tir.const(0, t0), tvm.tir.const(5, t1) - tvm.tir.const(4, t1)),
    )
    some_reducer1 = comm_reducer(
        lambda x, y: (
            x[0] + y[0],
            x[0] + y[0] + x[1] + y[1],
            x[0] * y[2] + y[0] * x[2],
            x[1] + y[2],
            4.0,
        ),
        lambda t0, t1, t2, t3, t4: (
            tvm.tir.const(0, t0),
            tvm.tir.const(1, t1),
            tvm.tir.const(2, t2),
            tvm.tir.const(3, t3),
            tvm.tir.const(4, t4),
        ),
    )

    k = te.reduce_axis((0, 10), name="k")
    A = te.placeholder((10,), name="A")
    # Test that SimplifyCombiner makes use of vranges
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(-10, -4))
    ck.verify(sum_or_prod(A[k], k), te.sum(A[k], k))
    ck.verify(sum_or_prod(A[k], k, init=1), te.sum(A[k], k, init=1))
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(5, 9), True)
    ck.verify(sum_or_prod(A[k], k), prod(A[k], k))
    ck.verify(sum_or_prod(A[k], k, init=1), prod(A[k], k, init=1))
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(-10, 100), True)
    ck.verify(sum_and_prod((A[k], A[10 - k]), k)[0], te.sum(A[k], k))
    ck.verify(sum_and_prod((A[k], A[10 - k]), k)[1], prod(A[10 - k], k))

    reference_simplified_sources = [
        [A[0]],
        [A[0], A[1]],
        [A[0], A[2]],
        [A[0], A[1], A[2], A[3]],
        [A[4]],
    ]
    for j in range(5):
        # Here we use the j-th component of the result, so only it and the components it
        # depends on are left.
        simplified = ck.analyzer.canonical_simplify(
            some_reducer1((A[0], A[1], A[2], A[3], A[4]), k)[j]
        )

        # Check that the remaining components are the expected ones.
        for lhs, rhs in zip(simplified.source, reference_simplified_sources[j]):
            assert tvm.ir.structural_equal(lhs, rhs)

    # Test that components with side effects are not removed
    dummy = tvm.ir.GlobalVar("dummy")
    side_effect = lambda *xs: tvm.tir.Call("int32", dummy, xs)
    ck.verify(
        sum_and_prod((A[k], side_effect(A[10 - k])), k)[0],
        sum_and_prod((A[k], side_effect(A[10 - k])), k)[0],
    )
    ck.verify(sum_and_prod((side_effect(A[k]), A[10 - k]), k)[0], te.sum(side_effect(A[k]), k))


def test_reduce_simplify():
    ck = CanonicalChecker()
    k = te.reduce_axis((0, 10), name="k")
    j = te.reduce_axis((-5, 3), name="j")
    A = te.placeholder((10,), name="A")
    ck.verify(te.sum(tvm.tir.Select(k + j < 12, k + j, 0), [k, j]), te.sum(k + j, [k, j]))
    ck.verify(te.sum(A[3], []), A[3])
    ck.verify(te.sum(A[3], [], where=k > 12, init=1.0), tvm.tir.const(1.0, dtype="float32"))
    # The rule below is not typical, removed for now
    ck.verify(te.sum(te.div(k, 10), k), te.sum(tvm.tir.const(0, "int32"), k))


def test_simplify_if_then_else():
    ck = CanonicalChecker()
    x = te.var("x")
    y = te.var("y")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    # simplification that takes condition into account.
    res = tvm.tir.if_then_else(
        (x * 4 + y) >= 466036,
        tvm.tir.if_then_else(
            24512 <= tmod(((x * 4) + y) - 466036, 24528),
            tmod(tmod(((x * 4) + y) - 466036, 24528) - 24512, 16),
            x,
        ),
        y,
    )

    res2 = tvm.tir.if_then_else(
        (x * 4) >= 466036 - y,
        tvm.tir.if_then_else(
            24512 <= tmod(((x * 4) + y) - 466036, 24528),
            tmod(tmod(((x * 4) + y) - 466036, 24528) - 24512, 16),
            x,
        ),
        y,
    )
    expected = tvm.tir.if_then_else(
        tvm.tir.LE(466036, (x * 4 + y)),
        tvm.tir.if_then_else(
            tvm.tir.LE(24512, tmod(((x * 4) + y) - 4, 24528)), tmod(((x * 4) + y) - 4, 16), x
        ),
        y,
    )
    ck.verify(res, expected)
    ck.verify(res2, expected)
    # can only simplify if condition
    res = tvm.tir.Select(tvm.tir.all(x >= -1, y >= 0), tmod(x + y + 100, 3), tmod(x + 100, 3))
    expected = tvm.tir.Select(tvm.tir.all(x >= -1, y >= 0), tmod(x + y + 1, 3), tmod(x + 100, 3))
    ck.verify(res, ck.analyzer.canonical_simplify(expected))

    res = tvm.tir.Select(x >= 10, tvm.tir.if_then_else(tdiv(x, 3) > 2, x, 0), 0)
    expected = tvm.tir.Select(x >= 10, x, 0)
    ck.verify(res, ck.analyzer.canonical_simplify(expected))

    res = tvm.tir.Select(x >= 10, tvm.tir.if_then_else(tdiv(x, 3) < 2, x, 0), 0)
    ck.verify(res, 0)


def test_complex_cases():
    ck = CanonicalChecker()
    x = te.var("x")
    y = te.var("y")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod
    res2 = (
        tdiv(tdiv(tmod(x * 128 + y, 1296), 36) * 2 + 1, 2) * 36
        + tdiv(tmod((x * 128) + y, 36) * 2 + 1, 2)
        - tmod((x * 128) + y, 1296)
        + 1
    )
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 5))
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 127))
    ck.verify(res2, 1)

    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1024), True)
    res3 = (
        tdiv(x * 1024 + y, 65536)
        + tdiv(tmod(x * 1024 + y, 65536), 256)
        + tdiv(tmod(x * 1024 + y, 256), 16)
        + tmod(x * 1024 + y, 16)
        - tdiv(y, 256)
        - tdiv(tmod(y, 256), 16)
        - tmod(y, 16)
        - (x * 4)
    )
    ck.verify(res3, tdiv((x * 1024) + y, 256) - tdiv(y, 256) - (x * 4))


def test_simplify_cast():
    ck = CanonicalChecker()
    tcast = tvm.tir.Cast
    fld = tvm.te.floordiv
    flm = tvm.te.floormod
    # cast(i64, i + j + 1) - cast(i64, i)
    i = te.var("i", dtype="int32")
    j = te.var("j", dtype="int32")
    res = tcast("int64", i + j + 1) - tcast("int64", i)
    ck.verify(res, tcast("int64", j) + tvm.tir.const(1, "int64"))
    # cast(i32, i + j + 1) - cast(i32, i)
    i = te.var("i", dtype="int64")
    j = te.var("j", dtype="int64")
    ck.analyzer.update(i, tvm.arith.ConstIntBound(0, 10))
    ck.analyzer.update(j, tvm.arith.ConstIntBound(0, 10))
    res = tcast("int32", i + j + 1) - tcast("int32", i)
    ck.verify(res, tcast("int32", j) + 1)
    # cast(i32, i + j - 100)
    i = te.var("i", dtype="int64")
    j = te.var("j", dtype="int64")
    ck.analyzer.update(i, tvm.arith.ConstIntBound(0, 2**31 - 1))
    ck.analyzer.update(j, tvm.arith.ConstIntBound(0, 10))
    res = tcast("int32", i + j - 100)
    ck.verify(res, res)
    # cast(i32, flm(axis, 7i64) * 2i64 + 1i64) + 1i32
    # - cast(i32, flm(axis, 7i64) * 2i64)
    axis = te.var("axis", dtype="int64")
    ck.analyzer.update(axis, tvm.arith.ConstIntBound(0, 42))
    res = (
        tcast(
            "int32",
            flm(axis, tvm.tir.const(7, "int64")) * tvm.tir.const(2, "int64")
            + tvm.tir.const(1, "int64"),
        )
        + tvm.tir.const(1, "int32")
        - tcast("int32", flm(axis, tvm.tir.const(7, "int64")) * tvm.tir.const(2, "int64"))
    )
    ck.verify(res, 2)


if __name__ == "__main__":
    test_floormod_simplify()
    test_mul_sum_simplify()
    test_simplify_if_then_else()
    test_div_simplify()
    test_reduce_simplify()
    test_reduce_combiner_simplify()

    test_split_index_simplify()
    test_canonical_mixed()
    test_complex_cases()
    test_simplify_cast()
