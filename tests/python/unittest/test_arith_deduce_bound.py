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
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.tir.buffer import decl_buffer


def test_deduce():
    a = te.var("a")
    b = te.var("b")
    c = te.var("c")
    d = te.var("d")

    b_s = tvm.arith.IntervalSet(2, 3)
    c_s = tvm.arith.IntervalSet(10, 15)
    d_s = tvm.arith.IntervalSet(-3, -1)
    zero = tvm.tir.const(0, "int32")

    fdiv = tvm.te.floordiv

    e0 = (-b) * a + c - d
    res0 = tvm.arith.deduce_bound(a, e0 >= 0, {b: b_s, c: c_s, d: d_s}, {})
    ans0 = fdiv(d - c, b * -1)
    tvm.testing.assert_prim_expr_equal(res0.max_value, ans0)

    # expression containing variable a is on rhs
    res0 = tvm.arith.deduce_bound(a, zero <= e0, {b: b_s, c: c_s, d: d_s}, {})
    tvm.testing.assert_prim_expr_equal(res0.max_value, ans0)

    e0 = d * a + c - d
    res0 = tvm.arith.deduce_bound(a, e0 >= 0, {b: b_s, c: c_s, d: d_s}, {})
    ans0 = fdiv(d - c, d)
    tvm.testing.assert_prim_expr_equal(res0.max_value, ans0)

    # expression containing variable a is on rhs
    res0 = tvm.arith.deduce_bound(a, zero <= e0, {b: b_s, c: c_s, d: d_s}, {})
    tvm.testing.assert_prim_expr_equal(res0.max_value, ans0)

    e1 = a * 4 + b < c
    res1 = tvm.arith.deduce_bound(a, e1, {b: b_s, c: c_s, d: d_s}, {})
    ans1 = fdiv(c - 1 - b, 4)
    tvm.testing.assert_prim_expr_equal(res1.max_value, ans1)

    # expression containing variable a is on rhs
    e1 = c > a * 4 + b
    res1 = tvm.arith.deduce_bound(a, e1, {b: b_s, c: c_s, d: d_s}, {})
    tvm.testing.assert_prim_expr_equal(res1.max_value, ans1)

    e2 = tvm.te.max(5, a * 4) < 0
    res2 = tvm.arith.deduce_bound(a, e2, {b: b_s, c: c_s, d: d_s}, {})
    assert str(res2.max_value) == "neg_inf"
    assert str(res2.min_value) == "pos_inf"

    # expression containing variable a is on rhs
    e2 = zero < tvm.te.max(5, a * 4)
    res2 = tvm.arith.deduce_bound(a, e2, {b: b_s, c: c_s, d: d_s}, {})
    assert str(res2.max_value) == "neg_inf"
    assert str(res2.min_value) == "pos_inf"

    e3 = (-b) + a * c - d
    res3 = tvm.arith.deduce_bound(a, e3 >= 0, {b: b_s, c: c_s, d: d_s}, {b: b_s, d: d_s})
    ans3 = fdiv(2, c) + 1
    tvm.testing.assert_prim_expr_equal(res3.min_value, ans3)

    res3 = tvm.arith.deduce_bound(a, zero <= e3, {b: b_s, c: c_s, d: d_s}, {b: b_s, d: d_s})
    tvm.testing.assert_prim_expr_equal(res3.min_value, ans3)

    # tests for `EQ` op
    res4 = tvm.arith.deduce_bound(a, a == b, {}, {})
    tvm.testing.assert_prim_expr_equal(res4.max_value, b)
    tvm.testing.assert_prim_expr_equal(res4.min_value, b)

    # Unsatisfiable `EQ`, variable as one of the Operand
    res5 = tvm.arith.deduce_bound(a, (a == b), {b: b_s}, {b: b_s})
    assert str(res5.max_value) == "neg_inf"
    assert str(res5.min_value) == "pos_inf"

    # variable `a` on the RHS side
    res6 = tvm.arith.deduce_bound(a, 10 == a, {}, {})
    tvm.testing.assert_prim_expr_equal(res6.max_value, 10)
    tvm.testing.assert_prim_expr_equal(res6.min_value, 10)

    # Add, Sub in `EQ`
    e4 = (a - c) == (b + d)
    ans4 = b + d + c
    res7 = tvm.arith.deduce_bound(a, e4, {b: b_s, c: c_s, d: d_s}, {})
    tvm.testing.assert_prim_expr_equal(res7.max_value, ans4)
    tvm.testing.assert_prim_expr_equal(res7.min_value, ans4)

    # Satisfiable Mul in `EQ` with negative sign
    res8 = tvm.arith.deduce_bound(a, (5 * a == -10), {}, {})
    tvm.testing.assert_prim_expr_equal(res8.max_value, -2)
    tvm.testing.assert_prim_expr_equal(res8.min_value, -2)

    # Unsatisfiable Mul in `EQ`
    e5 = 4 * a == b
    res9 = tvm.arith.deduce_bound(a, e5, {b: b_s}, {})
    assert str(res9.max_value) == "neg_inf"
    assert str(res9.min_value) == "pos_inf"

    res10 = tvm.arith.deduce_bound(a, (b * a == b), {b: b_s}, {})
    # simplifier is now able to prove symbolic relation (b * a % b == 0)
    tvm.testing.assert_prim_expr_equal(res10.max_value, 1)
    tvm.testing.assert_prim_expr_equal(res10.min_value, 1)


def test_check():
    a = te.var("a")
    b = te.var("b")
    c = te.var("c")
    d = te.var("d")

    b_s = tvm.arith.IntervalSet(2, 3)
    c_s = tvm.arith.IntervalSet(5, 7)
    d_s = tvm.arith.IntervalSet(-3, -1)

    # no compare operator
    res1 = tvm.arith.deduce_bound(a, a + b, {b: b_s}, {})
    assert res1.is_nothing()

    # multiple compare operators
    res2 = tvm.arith.deduce_bound(a, (a + b > 3).astype(c.dtype) > c, {b: b_s, c: c_s}, {})
    assert res2.is_nothing()

    # multiple target variable
    res2 = tvm.arith.deduce_bound(a, a * 2 - a > b, {b: b_s}, {})
    assert res2.is_nothing()


def test_deduce_basic():
    def test_basic(a1, a2, coff):
        a = te.var("a")
        b = te.var("b")
        b_s = tvm.arith.IntervalSet(a1, a2)
        e0 = b + a * coff + 3

        res1 = tvm.arith.deduce_bound(a, e0 < 17, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal((x * coff + 3 + y) < 17, True)

        # expression containing variable a is on rhs
        res1 = tvm.arith.deduce_bound(a, tvm.tir.const(17, "int32") < e0, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal((x * coff + 3 + y) > 17, True)

        # expression containing variable a is on rhs
        res1 = tvm.arith.deduce_bound(a, tvm.tir.const(17, "int32") >= e0, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]

        tvm.testing.assert_prim_expr_equal((x * coff + 3 + y) <= 17, True)

        res1 = tvm.arith.deduce_bound(a, e0 >= 17, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal((x * coff + 3 + y) >= 17, True)

    test_basic(0, 4, 4)
    test_basic(1, 5, 4)
    test_basic(2, 6, 4)
    test_basic(0, 4, -4)
    test_basic(1, 5, -4)
    test_basic(2, 6, -4)


def test_deduce_complex():
    def test_complex(a1, a2, coff):
        a = te.var("a")
        b = te.var("b")
        b_s = tvm.arith.IntervalSet(a1, a2)
        e0 = (b * 3 + a * coff) * 4

        res1 = tvm.arith.deduce_bound(a, e0 < 63, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal(((x * 3 + t * coff) * 4) < 63, True)

        # expression containing variable a is on rhs
        res1 = tvm.arith.deduce_bound(a, tvm.tir.const(63, "int32") >= e0, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal(((x * 3 + t * coff) * 4) <= 63, True)

        res1 = tvm.arith.deduce_bound(a, e0 > 63, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal(((x * 3 + t * coff) * 4) > 63, True)

        # expression containing variable a is on rhs
        res1 = tvm.arith.deduce_bound(a, tvm.tir.const(63, "int32") <= e0, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        tvm.testing.assert_prim_expr_equal(((x * 3 + t * coff) * 4) >= 63, True)

    test_complex(0, 4, 4)
    test_complex(0, 4, -4)
    test_complex(2, 6, 4)
    test_complex(0, 4, -4)
    test_complex(1, 5, -4)
    test_complex(2, 6, -4)


def test_deduce_non_support():
    a = te.var("a")

    def test_non_support(lhs):
        res = tvm.arith.deduce_bound(a, lhs < 10, {}, {})
        assert res.is_nothing()

    test_non_support(tvm.tir.floormod(a, 16))
    test_non_support(tvm.tir.Min(a, 16))
    test_non_support(tvm.tir.Max(a, 16))
    test_non_support(tvm.tir.LE(a, 16))
    test_non_support(tvm.tir.LT(a, 16))
    test_non_support(tvm.tir.GE(a, 16))
    test_non_support(tvm.tir.GT(a, 16))
    test_non_support(tvm.tir.EQ(a, 16))
    test_non_support(tvm.tir.NE(a, 16))
    test_non_support(tvm.tir.log(a))
    test_non_support(tvm.tir.BufferLoad(decl_buffer([16], "int32"), [a]))


def test_deduce_floordiv():
    def do_test(gen_expr, dom_map, expect_min, expect_max):
        a = te.var("a")
        expr = gen_expr(a)
        res = tvm.arith.deduce_bound(a, expr, dom_map, dom_map)
        if isinstance(expect_min, str):
            assert str(res.min_value) == expect_min
        else:
            tvm.testing.assert_prim_expr_equal(res.min_value, expect_min)
        if isinstance(expect_max, str):
            assert str(res.max_value) == expect_max
        else:
            tvm.testing.assert_prim_expr_equal(res.max_value, expect_max)

    # test basic cases
    do_test(lambda a: a // 8 > 3, {}, 32, "pos_inf")
    do_test(lambda a: a // 8 >= 3, {}, 24, "pos_inf")
    do_test(lambda a: a // 8 < 3, {}, "neg_inf", 23)
    do_test(lambda a: a // 8 <= 3, {}, "neg_inf", 31)
    do_test(lambda a: a // 8 == 3, {}, "pos_inf", "neg_inf")
    do_test(lambda a: a // 8 > -3, {}, -16, "pos_inf")
    do_test(lambda a: a // 8 >= -3, {}, -24, "pos_inf")
    do_test(lambda a: a // -8 > 3, {}, "neg_inf", -32)
    do_test(lambda a: a // -8 >= 3, {}, "neg_inf", -24)
    do_test(lambda a: a // -8 < 3, {}, -23, "pos_inf")
    do_test(lambda a: a // -8 <= 3, {}, -31, "pos_inf")
    do_test(lambda a: 8 // a >= 2, {}, "pos_inf", "neg_inf")

    # test nested cases
    b = te.var("b")
    bs = {b: tvm.arith.IntervalSet(2, 6)}
    do_test(lambda a: b * 3 + a // 8 < 63, bs, "neg_inf", 359)
    do_test(lambda a: b * 3 + a // 8 <= 63, bs, "neg_inf", 367)
    do_test(lambda a: b * 3 + a // 8 > 63, bs, 464, "pos_inf")
    do_test(lambda a: b * 3 + a // 8 >= 63, bs, 456, "pos_inf")


if __name__ == "__main__":
    tvm.testing.main()
