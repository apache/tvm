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


def assert_expr_equal(a, b):
    res =  tvm.ir_pass.Simplify(a - b)
    equal = isinstance(res, tvm.expr.IntImm) and res.value == 0
    if not equal:
        raise ValueError("{} and {} are not equal".format(a, b))


def test_deduce():
    a = tvm.var('a')
    b = tvm.var('b')
    c = tvm.var('c')
    d = tvm.var('d')

    b_s = tvm.arith.IntervalSet(2, 3)
    c_s = tvm.arith.IntervalSet(10, 15)
    d_s = tvm.arith.IntervalSet(-3, -1)
    zero = tvm.const(0, "int32")

    fdiv = tvm.floordiv

    e0 = (-b)*a+c-d
    res0 = tvm.arith.DeduceBound(a, e0>=0, {b: b_s, c: c_s, d: d_s}, {})
    ans0 = fdiv(d - c, b*-1)
    assert_expr_equal(res0.max_value, ans0)

    # expression containing variable a is on rhs
    res0 = tvm.arith.DeduceBound(a, zero <= e0, {b: b_s, c: c_s, d: d_s}, {})
    assert_expr_equal(res0.max_value, ans0)

    e0 = d*a+c-d
    res0 = tvm.arith.DeduceBound(a, e0>=0, {b: b_s, c: c_s, d: d_s}, {})
    ans0 = fdiv(d-c, d)
    assert_expr_equal(res0.max_value, ans0)

    # expression containing variable a is on rhs
    res0 = tvm.arith.DeduceBound(a, zero <= e0, {b: b_s, c: c_s, d: d_s}, {})
    assert_expr_equal(res0.max_value, ans0)


    e1 = (a*4+b < c)
    res1 = tvm.arith.DeduceBound(a, e1, {b: b_s, c: c_s, d: d_s}, {})
    ans1 = fdiv(c-1-b, 4)
    assert_expr_equal(res1.max_value, ans1)


    # expression containing variable a is on rhs
    e1 = (c > a*4+b)
    res1 = tvm.arith.DeduceBound(a, e1, {b: b_s, c: c_s, d: d_s}, {})
    assert_expr_equal(res1.max_value, ans1)


    e2 = (tvm.max(5, a * 4) < 0)
    res2 = tvm.arith.DeduceBound(a, e2, {b: b_s, c: c_s, d: d_s}, {})
    assert str(res2.max_value) == "neg_inf"
    assert str(res2.min_value) == "pos_inf"

    # expression containing variable a is on rhs
    e2 = (zero < tvm.max(5, a * 4))
    res2 = tvm.arith.DeduceBound(a, e2, {b: b_s, c: c_s, d: d_s}, {})
    assert str(res2.max_value) == "neg_inf"
    assert str(res2.min_value) == "pos_inf"

    e3 = (-b)+a*c-d
    res3 = tvm.arith.DeduceBound(a, e3>=0, {b: b_s, c: c_s, d: d_s}, {b: b_s, d: d_s})
    ans3 = fdiv(2,c)+1
    assert str(tvm.ir_pass.Simplify(res3.min_value)) == str(ans3)

    res3 = tvm.arith.DeduceBound(a, zero <= e3, {b: b_s, c: c_s, d: d_s}, {b: b_s, d: d_s})
    assert str(tvm.ir_pass.Simplify(res3.min_value)) == str(ans3)

    # tests for `EQ` op
    res4 = tvm.arith.DeduceBound(a, a == b, {}, {})
    assert_expr_equal(res4.max_value, b)
    assert_expr_equal(res4.min_value, b)

    # Unsatisfiable `EQ`, variable as one of the Operand
    res5 = tvm.arith.DeduceBound(a, (a == b), {b: b_s}, {b: b_s})
    assert str(res5.max_value) == "neg_inf"
    assert str(res5.min_value) == "pos_inf"

    # variable `a` on the RHS side
    res6 = tvm.arith.DeduceBound(a, 10 == a, {}, {})
    assert_expr_equal(res6.max_value, 10)
    assert_expr_equal(res6.min_value, 10)

    # Add, Sub in `EQ`
    e4 = ((a - c) == (b + d))
    ans4 = (b + d + c)
    res7 = tvm.arith.DeduceBound(a, e4, {b: b_s, c: c_s, d: d_s}, {})
    assert_expr_equal(res7.max_value, ans4)
    assert_expr_equal(res7.min_value, ans4)

    # Satisfiable Mul in `EQ` with negative sign
    res8 = tvm.arith.DeduceBound(a, (5 * a == -10), {}, {})
    assert_expr_equal(res8.max_value, -2)
    assert_expr_equal(res8.min_value, -2)

    # Unsatisfiable Mul in `EQ`
    e5 = (4 * a == b)
    res9 = tvm.arith.DeduceBound(a, e5, {b: b_s}, {})
    assert str(res9.max_value) == "neg_inf"
    assert str(res9.min_value) == "pos_inf"

    # Unsatisfiable Mul in `EQ`
    res10 = tvm.arith.DeduceBound(a, (b * a == b), {b: b_s}, {})    # simplifier is not able to prove that (b % b == 0)
    assert str(res10.max_value) == "neg_inf"
    assert str(res10.min_value) == "pos_inf"


def test_check():
    a = tvm.var('a')
    b = tvm.var('b')
    c = tvm.var('c')
    d = tvm.var('d')

    b_s = tvm.arith.IntervalSet(2, 3)
    c_s = tvm.arith.IntervalSet(5, 7)
    d_s = tvm.arith.IntervalSet(-3, -1)

    # no compare operator
    res1 = tvm.arith.DeduceBound(a, a+b, {b: b_s}, {})
    assert res1.is_nothing()

    # multiple compare operators
    res2 = tvm.arith.DeduceBound(a, (a+b>3).astype(c.dtype)>c , {b: b_s, c: c_s}, {})
    assert res2.is_nothing()

    # multiple target variable
    res2 = tvm.arith.DeduceBound(a, a*2-a>b, {b: b_s}, {})
    assert res2.is_nothing()

def test_deduce_basic():
    def test_basic(a1, a2, coff):
        a = tvm.var('a')
        b = tvm.var('b')
        b_s = tvm.arith.IntervalSet(a1, a2)
        e0 = b + a*coff + 3

        res1 = tvm.arith.DeduceBound(a, e0<17, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify((x * coff + 3 + y) < 17)).value == 1

        # expression containing variable a is on rhs
        res1 = tvm.arith.DeduceBound(a, tvm.const(17, "int32") < e0, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify((x * coff + 3 + y) > 17)).value == 1

        # expression containing variable a is on rhs
        res1 = tvm.arith.DeduceBound(a, tvm.const(17, "int32")>= e0, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify((x * coff + 3 + y) <= 17)).value == 1

        res1 = tvm.arith.DeduceBound(a, e0>=17, {b: b_s}, {b: b_s})
        [x, y] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify((x * coff + 3 + y) >= 17)).value == 1

    test_basic(0, 4, 4)
    test_basic(1, 5, 4)
    test_basic(2, 6, 4)
    test_basic(0, 4, -4)
    test_basic(1, 5, -4)
    test_basic(2, 6, -4)

def test_deduce_complex():
    def test_complex(a1, a2, coff):
        a = tvm.var('a')
        b = tvm.var('b')
        b_s = tvm.arith.IntervalSet(a1, a2)
        e0 = (b*3 + a* coff) * 4

        res1 = tvm.arith.DeduceBound(a, e0<63, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify(((x*3 + t* coff) * 4) < 63)).value == 1

        # expression containing variable a is on rhs
        res1 = tvm.arith.DeduceBound(a, tvm.const(63, "int32")>= e0, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff > 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify(((x*3 + t* coff) * 4) <= 63)).value == 1

        res1 = tvm.arith.DeduceBound(a, e0>63, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify(((x*3 + t* coff) * 4) > 63)).value == 1

        # expression containing variable a is on rhs
        res1 = tvm.arith.DeduceBound(a, tvm.const(63, "int32") <= e0, {b: b_s}, {b: b_s})
        [t, x] = [res1.max_value, b_s.max_value] if coff < 0 else [res1.min_value, b_s.min_value]
        assert (tvm.ir_pass.Simplify(((x*3 + t* coff) * 4) >= 63)).value == 1

    test_complex(0, 4, 4)
    test_complex(0, 4, -4)
    test_complex(2, 6, 4)
    test_complex(0, 4, -4)
    test_complex(1, 5, -4)
    test_complex(2, 6, -4)


if __name__ == "__main__":
    test_check()
    test_deduce()
    test_deduce_basic()
    test_deduce_complex()
