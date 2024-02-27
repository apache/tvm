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

import contextlib

import pytest

import tvm
import tvm.testing

from tvm import te
from tvm.arith import ConstIntBound

NEG_INF = ConstIntBound.NEG_INF
POS_INF = ConstIntBound.POS_INF


class TestCase:
    def __init__(self, expr, expected_bounds, known_bounds=None, constraint=None):
        self.expr = expr
        self.expected_bounds = expected_bounds
        if known_bounds is None:
            self.known_bounds = {}
        else:
            self.known_bounds = known_bounds

        self.constraint = constraint

    @property
    def __name__(self):
        return str(self.expr)


class BaseCompare:
    def test_const_bounds(self, test_case):
        analyzer = tvm.arith.Analyzer()

        for var, bounds in test_case.known_bounds.items():
            analyzer.update(var, ConstIntBound(*bounds))

        with contextlib.ExitStack() as stack:
            if test_case.constraint is not None:
                stack.enter_context(analyzer.constraint_scope(test_case.constraint))

            bounds = analyzer.const_int_bound(test_case.expr)

        if test_case.expected_bounds[0] is None:
            assert bounds.max_value == test_case.expected_bounds[1]
        elif test_case.expected_bounds[1] is None:
            assert bounds.min_value == test_case.expected_bounds[0]
        else:
            assert (bounds.min_value, bounds.max_value) == test_case.expected_bounds


class TestDataType(BaseCompare):
    test_case = tvm.testing.parameter(
        TestCase(te.var("x", dtype="int64"), (NEG_INF, POS_INF)),
        TestCase(te.var("x", dtype="int8"), (-128, 127)),
        TestCase(te.var("x", dtype="uint8"), (0, 255)),
        TestCase(te.size_var("x", dtype="int32"), (0, POS_INF)),
    )


class TestCastBound(BaseCompare):
    x = te.var("x", dtype="int8")
    tmod = tvm.tir.truncmod

    test_case = tvm.testing.parameter(
        TestCase(tmod(x, 3).astype("uint32"), (0, 2)),
        TestCase(tmod(x, 3).astype("float32").astype("int32"), (-2, 2)),
    )


class TestAddSubBound(BaseCompare):
    x = te.var("x", "int64")
    y = te.var("y", "int64")

    test_case = tvm.testing.parameter(
        TestCase(x + y, (NEG_INF, POS_INF)),
        TestCase(x + y, (1, 14), known_bounds={x: (0, 4), y: (1, 10)}),
        TestCase(x - y, (-10, 3), known_bounds={x: (0, 4), y: (1, 10)}),
        TestCase(x - y, (-10, POS_INF), known_bounds={x: (0, POS_INF), y: (1, 10)}),
        TestCase(1 - x, (NEG_INF, 1), known_bounds={x: (0, POS_INF), y: (1, 10)}),
    )


@pytest.mark.xfail(reason="Not currently supported")
class TestBoundsUsingReciprocals(BaseCompare):
    """Special handling for differences of reciprocals

    These terms can appear when comparing the number of operations for
    different orderings of matrix multiplications, with A, B, and C
    known to be positive values.

    In these cases, comparing `(A+B)*C < A*B` is equivalent to
    `1/A + 1/B < 1/C`.  Working in terms of the reciprocals
    allows the ConstIntBound analyzer to provide a tighter
    bound for these differences than would otherwise be
    available.

    For `(A+B)*C - A*B`, the normal bottom-up integer bounds are unable to
    provide the bounds required to provide these inequalities, because they
    treat the terms as uncorrelated.  That is, they assume that `(A+B)*C` may
    achieve its minimum while `A*B` simultaneously achieves its maximum.
    """

    A, B, C = [te.var(letter, "int64") for letter in "ABC"]

    symmetric_bounds = {A: (1, 4095), B: (1, 4095), C: (2048, 2048)}
    asymmetric_bounds = {A: (1, 1024), B: (1, POS_INF), C: (2048, 2048)}

    test_case = tvm.testing.parameter(
        TestCase((A + B) * C - A * B, (2048, None), known_bounds=symmetric_bounds),
        TestCase((A + B) * C - B * A, (2048, None), known_bounds=symmetric_bounds),
        TestCase(A * B - (A + B) * C, (None, -2048), known_bounds=symmetric_bounds),
        TestCase(B * A - (A + B) * C, (None, -2048), known_bounds=symmetric_bounds),
        TestCase((A + B) * C - A * B, (2048, None), known_bounds=asymmetric_bounds),
        TestCase((A + B) * C - B * A, (2048, None), known_bounds=asymmetric_bounds),
        TestCase(A * B - (A + B) * C, (None, -2048), known_bounds=asymmetric_bounds),
        TestCase(B * A - (A + B) * C, (None, -2048), known_bounds=asymmetric_bounds),
    )


class TestMulBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    test_case = tvm.testing.parameter(
        TestCase(x * y + 20, (0, 60), {x: (-2, 4), y: (4, 10)}),
        TestCase(x * y, (-32, 24), {x: (-3, 4), y: (-8, 2)}),
        TestCase(x * y, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-8, 2)}),
    )


class TestTruncDivBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    expr = tvm.tir.truncdiv(x, y)

    test_case = tvm.testing.parameter(
        TestCase(expr, (-2, None), {x: (-9, 4), y: (4, 10)}),
        TestCase(expr, (-4, 9), {x: (-9, 4), y: (-2, 0)}),
        TestCase(expr, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-2, 1)}),
        TestCase(expr, (-9, 9), {x: (-9, 4), y: (-4, 12)}),
    )


class TestTruncModBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    expr = tvm.tir.truncmod(x, y)

    test_case = tvm.testing.parameter(
        TestCase(expr, (-9, 4), {x: (-9, 4), y: (4, 10)}),
        TestCase(expr, (-9, 9), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(expr, (0, 9), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestFloorDivBound(BaseCompare):
    x, y = te.var("x"), te.var("y")
    ux = te.var("x", dtype="uint32")
    uy = te.var("y", dtype="uint32")

    test_case = tvm.testing.parameter(
        TestCase(x // y, (-9 // 4, None), {x: (-9, 4), y: (4, 10)}),
        TestCase(x // y, (-4, 9), {x: (-9, 4), y: (-2, 0)}),
        TestCase(x // y, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-2, 1)}),
        TestCase(x // y, (-9, 9), {x: (-9, 4), y: (-4, 12)}),
        TestCase(ux // uy, (0, 4), {ux: (1, 4), uy: (0, 12)}),
    )


class TestFloorModBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    test_case = tvm.testing.parameter(
        TestCase(x % y, (0, 9), {x: (-9, 4), y: (4, 10)}),
        TestCase(x % y, (0, 9), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(x % y, (0, 9), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestMinMaxBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    test_case = tvm.testing.parameter(
        TestCase(tvm.te.min(x, y), (-9, 10), {x: (-9, 11), y: (4, 10)}),
        TestCase(tvm.te.min(x, y), (NEG_INF, 10), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(tvm.te.max(x, y), (4, POS_INF), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(tvm.te.max(x, y), (4, POS_INF), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestSelectBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    test_case = tvm.testing.parameter(
        TestCase(
            tvm.tir.Select(x > 1, (y < 0).astype("int32"), y + 1),
            (0, 11),
            {x: (-9, 11), y: (4, 10)},
        ),
    )


class TestShiftAndBound(BaseCompare):
    x, y = te.var("x"), te.var("y")

    test_case = tvm.testing.parameter(
        TestCase(x >> y, (-3, 2), {x: (-9, 11), y: (2, 10)}),
        TestCase(x & y, (0, 10), {x: (-9, 11), y: (2, 10)}),
        TestCase(x & y, (0, 10), {x: (10, 11), y: (2, 10)}),
    )


class TestMixIndexBound(BaseCompare):
    x, y = te.var("x"), te.var("y")
    tdiv = tvm.tir.truncdiv
    tmod = tvm.tir.truncmod

    test_case = tvm.testing.parameter(
        TestCase(tmod(x, 8) + tdiv(x, 8) * 8, (0, 24 - 1), {x: (0, 24 - 1), y: (0, 3 - 1)}),
        TestCase(y + x * 3, (0, 24 * 3 - 1), {x: (0, 24 - 1), y: (0, 3 - 1)}),
        TestCase(
            tmod(x, 7) + tdiv(x, 7) * 7, (0, (23 // 7) * 7 + 6), {x: (0, 24 - 1), y: (0, 3 - 1)}
        ),
    )


class TestLetBound(BaseCompare):
    x = te.var("x")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.Let(x, 1, x + 1), (2, 2)),
    )


class TestFloorModNegativeDivisor(BaseCompare):
    flm, fld = tvm.te.floormod, tvm.te.floordiv
    a, b = te.var("a"), te.var("b")

    test_case = tvm.testing.parameter(
        TestCase(a % b, (-4, 6), {a: (0, 6), b: (-5, 7)}),
    )


class TestDivModAssumeNoZeroDivisor(BaseCompare):
    """Divmod non negative expression makes assumption that divide by
    zero won't occur this assumption is important to get best result
    from symbolic shape programs
    """

    a, b = te.var("a"), te.var("b")

    test_case = tvm.testing.parameter(
        TestCase(a // b, (0, 6), {a: (0, 6), b: (0, POS_INF)}),
        TestCase(a % b, (0, 6), {a: (0, 6), b: (0, POS_INF)}),
    )


class TestMultipleCondition(BaseCompare):
    a = te.var("a")
    test_case = tvm.testing.parameter(
        TestCase(
            a % 58 - 1,
            (0, None),
            known_bounds={a: (0, 128)},
            constraint=tvm.tir.all(1 <= a % 58, a % 58 < 57),
        ),
    )


class TestBroadcastBound(BaseCompare):
    a = te.var("a")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.Broadcast(a, 4), (0, 128), {a: (0, 128)}),
    )


class TestRampBound(BaseCompare):
    a = te.var("a")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tir.Ramp(a, 2, 4) + 2, (2, 128 + 2 * 3 + 2), {a: (0, 128)}),
    )


if __name__ == "__main__":
    tvm.testing.main()
