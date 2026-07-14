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
# ruff: noqa: RUF012

import contextlib

import pytest

import tvm
import tvm.testing
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
            assert analyzer.const_int_bound_is_bound(var)

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
        TestCase(tvm.tirx.Var("x", "int64"), (NEG_INF, POS_INF)),
        TestCase(tvm.tirx.Var("x", "int8"), (-128, 127)),
        TestCase(tvm.tirx.Var("x", "uint8"), (0, 255)),
        TestCase(tvm.tirx.Var("x", "int32"), (-(2**31), 2**31 - 1)),
    )


def test_plain_var_non_negative_bound_requires_context():
    var = tvm.tirx.Var("x", "int64")
    analyzer = tvm.arith.Analyzer()

    assert analyzer.const_int_bound(var).min_value == NEG_INF
    with analyzer.constraint_scope(var >= 0):
        assert analyzer.const_int_bound(var).min_value == 0
    assert analyzer.const_int_bound(var).min_value == NEG_INF


class TestCastBound(BaseCompare):
    x = tvm.tirx.Var("x", "int8")
    tmod = tvm.tirx.truncmod

    test_case = tvm.testing.parameter(
        TestCase(tmod(x, 3).astype("uint32"), (0, 2)),
        TestCase(tmod(x, 3).astype("float32").astype("int32"), (-2, 2)),
    )


class TestAddSubBound(BaseCompare):
    x = tvm.tirx.Var("x", "int64")
    y = tvm.tirx.Var("y", "int64")

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

    A, B, C = [tvm.tirx.Var(letter, "int64") for letter in "ABC"]

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
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    test_case = tvm.testing.parameter(
        TestCase(x * y + 20, (0, 60), {x: (-2, 4), y: (4, 10)}),
        TestCase(x * y, (-32, 24), {x: (-3, 4), y: (-8, 2)}),
        TestCase(x * y, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-8, 2)}),
    )


class TestTruncDivBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    expr = tvm.tirx.truncdiv(x, y)

    test_case = tvm.testing.parameter(
        TestCase(expr, (-2, None), {x: (-9, 4), y: (4, 10)}),
        TestCase(expr, (-4, 9), {x: (-9, 4), y: (-2, 0)}),
        TestCase(expr, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-2, 1)}),
        TestCase(expr, (-9, 9), {x: (-9, 4), y: (-4, 12)}),
    )


class TestTruncModBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    expr = tvm.tirx.truncmod(x, y)

    test_case = tvm.testing.parameter(
        TestCase(expr, (-9, 4), {x: (-9, 4), y: (4, 10)}),
        TestCase(expr, (-9, 9), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(expr, (0, 9), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestFloorDivBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")
    ux = tvm.tirx.Var("x", "uint32")
    uy = tvm.tirx.Var("y", "uint32")

    test_case = tvm.testing.parameter(
        TestCase(x // y, (-9 // 4, None), {x: (-9, 4), y: (4, 10)}),
        TestCase(x // y, (-4, 9), {x: (-9, 4), y: (-2, 0)}),
        TestCase(x // y, (NEG_INF, POS_INF), {x: (NEG_INF, 4), y: (-2, 1)}),
        TestCase(x // y, (-9, 9), {x: (-9, 4), y: (-4, 12)}),
        TestCase(ux // uy, (0, 4), {ux: (1, 4), uy: (0, 12)}),
    )


class TestFloorModBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    test_case = tvm.testing.parameter(
        TestCase(x % y, (0, 9), {x: (-9, 4), y: (4, 10)}),
        TestCase(x % y, (0, 9), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(x % y, (0, 9), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestModBoundWithModularSet(BaseCompare):
    """floormod/truncmod bounds tightened by modular-set information.

    When the dividend satisfies `a == base (mod coeff)` and
    `g = gcd(coeff, divisor) > 1`, `floormod(a, divisor)` can only take the
    values `{r, r + g, ..., divisor - g + r}` where `r = base % g`.

    Regression test for a bug where the residue was normalized modulo the
    divisor instead of modulo `g`, yielding invalid bounds (min > max) such
    as [255, 191] for `(n * 320 + 255) % 256`. Such bounds let
    `CanProve(..., kSymbolicBound)` incorrectly validate the bounds
    predicates of imperfect loop splits, so scheduled GPU kernels silently
    lost their out-of-bounds guards.
    """

    n = tvm.tirx.Var("n", "int64")
    tmod = tvm.tirx.truncmod

    test_case = tvm.testing.parameter(
        # gcd(320, 256) = 64, base 255 -> residue 63: values {63, 127, 191, 255}
        TestCase((n * 320 + 255) % 256, (63, 255)),
        # coeff divides the divisor, base 0: multiples of 16
        TestCase((n * 16) % 7168, (0, 7152)),
        # base already smaller than the gcd: values {3, 67, 131, 195}
        TestCase((n * 64 + 3) % 256, (3, 195)),
        # truncated mod mirrors the residues on the negative side
        TestCase(tmod(n * 64 + 3, 256), (-253, 195)),
        # non-negative dividend keeps the one-sided range
        TestCase(tmod(n * 64 + 3, 256), (3, 195), {n: (0, POS_INF)}),
        # the modular bound must not discard a tighter interval bound:
        # dividend in [63, 127] -> values {63, 127}, not [63, 255]
        TestCase((n * 64 + 63) % 256, (63, 127), {n: (0, 1)}),
        # same for truncmod with a negative dividend range: values {-67, -3}
        TestCase(tmod(n * 64 + 61, 256), (-67, -3), {n: (-2, -1)}),
        # floormod of the same negative range: values {189, 253}, the
        # modular residue set {61, 125, 189, 253} bounds it to [61, 253]
        TestCase((n * 64 + 61) % 256, (61, 253), {n: (-2, -1)}),
        # Truncated mod with an entirely-negative dividend whose magnitude is
        # below the divisor: no reduction happens, so the result equals the
        # dividend and the bound is [a.min, a.max], not the loose [a.min, 0].
        TestCase(tmod(n, 256), (-5, -3), {n: (-5, -3)}),
        # A negative dividend that spans a multiple of the divisor can still
        # reach 0, so the upper bound stays 0 (no tightening here).
        TestCase(tmod(n, 256), (-255, 0), {n: (-1000, -300)}),
    )


class TestMinMaxBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    test_case = tvm.testing.parameter(
        TestCase(tvm.tirx.min(x, y), (-9, 10), {x: (-9, 11), y: (4, 10)}),
        TestCase(tvm.tirx.min(x, y), (NEG_INF, 10), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(tvm.tirx.max(x, y), (4, POS_INF), {x: (NEG_INF, POS_INF), y: (4, 10)}),
        TestCase(tvm.tirx.max(x, y), (4, POS_INF), {x: (1, POS_INF), y: (4, 10)}),
    )


class TestSelectBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    test_case = tvm.testing.parameter(
        TestCase(
            tvm.tirx.Select(x > 1, (y < 0).astype("int32"), y + 1),
            (0, 11),
            {x: (-9, 11), y: (4, 10)},
        ),
    )


class TestShiftAndBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")

    test_case = tvm.testing.parameter(
        TestCase(x >> y, (-3, 2), {x: (-9, 11), y: (2, 10)}),
        TestCase(x & y, (0, 10), {x: (-9, 11), y: (2, 10)}),
        TestCase(x & y, (0, 10), {x: (10, 11), y: (2, 10)}),
    )


class TestMixIndexBound(BaseCompare):
    x, y = tvm.tirx.Var("x", "int32"), tvm.tirx.Var("y", "int32")
    tdiv = tvm.tirx.truncdiv
    tmod = tvm.tirx.truncmod

    test_case = tvm.testing.parameter(
        TestCase(tmod(x, 8) + tdiv(x, 8) * 8, (0, 24 - 1), {x: (0, 24 - 1), y: (0, 3 - 1)}),
        TestCase(y + x * 3, (0, 24 * 3 - 1), {x: (0, 24 - 1), y: (0, 3 - 1)}),
        TestCase(
            tmod(x, 7) + tdiv(x, 7) * 7, (0, (23 // 7) * 7 + 6), {x: (0, 24 - 1), y: (0, 3 - 1)}
        ),
    )


class TestLetBound(BaseCompare):
    x = tvm.tirx.Var("x", "int32")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tirx.Let(x, 1, x + 1), (2, 2)),
    )


class TestFloorModNegativeDivisor(BaseCompare):
    flm, fld = tvm.tirx.floormod, tvm.tirx.floordiv
    a, b = tvm.tirx.Var("a", "int32"), tvm.tirx.Var("b", "int32")

    test_case = tvm.testing.parameter(
        TestCase(a % b, (-4, 6), {a: (0, 6), b: (-5, 7)}),
    )


class TestDivModAssumeNoZeroDivisor(BaseCompare):
    """Divmod non negative expression makes assumption that divide by
    zero won't occur this assumption is important to get best result
    from symbolic shape programs
    """

    a, b = tvm.tirx.Var("a", "int32"), tvm.tirx.Var("b", "int32")

    test_case = tvm.testing.parameter(
        TestCase(a // b, (0, 6), {a: (0, 6), b: (0, POS_INF)}),
        TestCase(a % b, (0, 6), {a: (0, 6), b: (0, POS_INF)}),
    )


class TestMultipleCondition(BaseCompare):
    a = tvm.tirx.Var("a", "int32")
    test_case = tvm.testing.parameter(
        TestCase(
            a % 58 - 1,
            (0, None),
            known_bounds={a: (0, 128)},
            constraint=tvm.tirx.all(1 <= a % 58, a % 58 < 57),
        ),
    )


class TestBroadcastBound(BaseCompare):
    a = tvm.tirx.Var("a", "int32")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tirx.Broadcast(a, 4), (0, 128), {a: (0, 128)}),
    )


class TestRampBound(BaseCompare):
    a = tvm.tirx.Var("a", "int32")
    test_case = tvm.testing.parameter(
        TestCase(tvm.tirx.Ramp(a, 2, 4) + 2, (2, 128 + 2 * 3 + 2), {a: (0, 128)}),
    )


class TestModularSetBound(BaseCompare):
    analyzer = tvm.arith.Analyzer()
    tx = tvm.tirx.Var("tx", "int32")
    bx = tvm.tirx.Var("bx", "int32")

    expr = (bx * 2048 + tx * 16) % 7168

    test_case = tvm.testing.parameter(
        TestCase(expr, (0, 7152), {bx: (0, 3584), tx: (0, 128)}),
    )


if __name__ == "__main__":
    tvm.testing.main()
