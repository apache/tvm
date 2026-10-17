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

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_stmt_simplify():
    @T.function(private=True)
    def before(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((10,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((10,), "float32", data=C)])
        n_val: T.let[T.int32] = 10
        for i in T.serial(n_val):
            if i < 12:
                A_ptr[i] = C_ptr[i]

    @T.function(private=True)
    def expected(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((10,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((10,), "float32", data=C)])
        n_val: T.let[T.int32] = 10
        for i in T.serial(10):
            A_ptr[i] = C_ptr[i]

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_thread_extent_simplify():
    @T.function(private=True)
    def before(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((10,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((10,), "float32", data=C)])
        n_val: T.let[T.int32] = 10
        with T.launch_thread("threadIdx.x", n_val) as tx:
            with T.launch_thread("threadIdx.y", 1) as ty:
                if tx + ty < 12:
                    A_ptr[tx] = C_ptr[tx + ty]

    @T.function(private=True)
    def expected(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((10,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((10,), "float32", data=C)])
        n_val: T.let[T.int32] = 10
        with T.launch_thread("threadIdx.x", 10) as tx:
            with T.launch_thread("threadIdx.y", 1) as ty:
                A_ptr[tx] = C_ptr[tx]

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_if_likely():
    @T.function(private=True)
    def before(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((32,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((1024,), "float32", data=C)])
        with T.launch_thread("threadIdx.x", 32) as tx:
            with T.launch_thread("threadIdx.y", 32) as ty:
                if T.likely(tx * 32 + ty < n):
                    if T.likely(tx * 32 + ty < n):
                        A_ptr[tx] = C_ptr[tx * 32 + ty]

    @T.function(private=True)
    def expected(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_tensor(A, ty_args=[T.Tensor((32,), "float32", data=A)])
        C_ptr = T.decl_tensor(C, ty_args=[T.Tensor((1024,), "float32", data=C)])
        with T.launch_thread("threadIdx.x", 32) as tx:
            with T.launch_thread("threadIdx.y", 32) as ty:
                if T.likely(tx * 32 + ty < n):
                    A_ptr[tx] = C_ptr[tx * 32 + ty]

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_loop_body_knows_dynamic_extent_is_positive():
    @T.function(private=True)
    def before(A: T.Tensor((1,), "float32"), m: T.int32, n: T.int32):
        for i in T.serial(m, n // 4):
            if n // 4 - m > 0:
                A[0] = 1.0

    @T.function(private=True)
    def expected(A: T.Tensor((1,), "float32"), m: T.int32, n: T.int32):
        for i in T.serial(m, n // 4):
            A[0] = 1.0

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    after = tvm_tirx.tirx.transform.StmtSimplify()(InputModule)["main"]
    tvm_tirx.ir.assert_structural_equal(after, expected)


def _apply_simplify(func):
    """Helper to apply the default statement simplification."""
    input_func = func

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    return tvm_tirx.tirx.transform.StmtSimplify()(InputModule)["main"]


def test_load_store_noop():
    """Store of a value that was just read from the same location is a no-op."""

    @T.function(private=True)
    def before(A: T.Tensor((1,), "float32")):
        A[0] = A[0]

    @T.function(private=True)
    def expected(A: T.Tensor((1,), "float32")):
        T.evaluate(0)

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_load_store_noop_after_simplify():
    """As test_load_store_noop, but requiring simplification to identify.

    Previously, a bug caused the self-assignment of a tensor to
    checked based on the pre-simplification assignment, not the
    post-simplification.  This test is to identify any similar
    regression.
    """

    @T.function(private=True)
    def before(A: T.Tensor((1,), "float32")):
        A[0] = A[0] + (5.0 - 5.0)

    @T.function(private=True)
    def expected(A: T.Tensor((1,), "float32")):
        T.evaluate(0)

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_nested_condition():
    """Nested If with the same condition can be simplified.

    Requires const_int_bound to narrow scope of i within the
    conditional, or for rewrite_simplify to recognize the literal
    constraint.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_nested_provable_condition():
    """Simplify inner conditional using constraint from outer.

    Requires const_int_bound to narrow scope of i within the
    conditional.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i < 7:
                    A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_nested_var_condition():
    """Simplify inner conditional using constraint from outer.

    Requires for rewrite_simplify to recognize the repeated
    constraint.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i == n:
                    A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_altered_tensor_contents():
    """No simplification of data-dependent conditionals.

    A literal constraint must not be propagated if the values
    referenced may change.  TIR requires single assignment of
    variables, so Var objects may be assumed constant, but TensorLoad
    may not.
    """

    @T.function(private=True)
    def before(A: T.Tensor((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0

    expected = before

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_negation_of_condition():
    """Use negation of outer condition to simplify innner.

    Within the body of an if statement, the negation of the
    condition is known to be false.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                if i != 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 1

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_negation_of_not_equal():
    """As test_negation_of_var_condition, but with a != outer condition.

    Because ConstIntBoundAnalyzer only tracks the min and max allowed
    values, the outer i!=5 condition does provide a constraint on the
    bounds.  This test relies on RewriteSimplifier to recognize
    ``i==5`` as the negation of a literal constraint.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                if i == 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                A[i] = 1

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_negation_of_var_condition():
    """As test_negation_of_var_condition, but with a dynamic condition.

    This simplification cannot be done with ConstIntBoundAnalyzer, and
    must rely on RewriteSimplifier recognizing the repeated literal.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i != n:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.function(private=True)
    def expected(A: T.Tensor((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 1

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_literal_constraint_split_boolean_and():
    """Split a boolean AND into independent constraints

    A single if condition may impose multiple literal constraints.
    Each constraint that is ANDed together to form the condition
    should be treated as an independent constraint.  The use of n in
    the condition is to ensure we exercise RewriteSimplifier.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                if i == n:
                    A[i, j] = 0

    @T.function(private=True)
    def expected(A: T.Tensor((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                A[i, j] = 0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_literal_constraint_split_boolean_or():
    """Split a boolean OR into independent constraints

    Similar to test_literal_constraint_split_boolean_and, but splitting a
    boolean OR into independent conditions.  This uses the
    simplification that ``!(x || y) == !x && !y``.

    The use of ``n`` in the condition is to ensure we exercise
    RewriteSimplifier.
    """

    @T.function(private=True)
    def before(A: T.Tensor((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2

    @T.function(private=True)
    def expected(A: T.Tensor((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                A[i, j] = 2

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_prove_condition_using_let():
    """Simplify conditions using non-inlined let bindings

    Not all let bindings are inlined when they occur in later
    expressions.  However, even if they are not inlined, they may be
    used to prove the value of a condition.

    With flat Bind, the analyzer binds the variable to its value for
    constraint proving, which also substitutes the variable in later
    expressions.
    """

    @T.function(private=True)
    def before(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if condition or i >= 3:
                A[i] = condition

    @T.function(private=True)
    def expected(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            A[i] = i < 3

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_prove_let_condition():
    """Simplify conditions using non-inlined let bindings

    With flat Bind, analyzer binds variable to value, which also
    substitutes the variable in later expressions.
    """

    @T.function(private=True)
    def before(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if i < 3:
                if condition:
                    A[i] = condition

    @T.function(private=True)
    def expected(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if i < 3:
                A[i] = T.bool(True)

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_prove_repeated_let_condition():
    """Simplify conditions using non-inlined let bindings

    With analyzer Bind, the variable is substituted with its value,
    so `if condition` becomes `if i < 3`, and within that context
    the inner `if condition` simplifies to True and is eliminated.
    """

    @T.function(private=True)
    def before(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if condition:
                if condition:
                    A[i] = condition

    @T.function(private=True)
    def expected(A: T.Tensor(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if i < 3:
                A[i] = T.bool(True)

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_if_then_else_expr():
    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0)

    @T.function(private=True)
    def expected(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_ceil_log2_int():
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.function(private=True)
    def before(A: T.Tensor(1, "int32")):
        A[0] = T.cast(T.ceil(T.log2(T.cast(14, ty="float64"))), ty="int32")

    @T.function(private=True)
    def expected(A: T.Tensor(1, "int32")):
        A[0] = 4

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_left_ceil_log2_lower_bound():
    """Integer bounds are propagated through topi.math.ceil_log2

    With flat Bind, the Bind is kept even when the variable is unused
    after simplification. The if condition is still eliminated.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            x: T.let[T.int32] = T.cast(
                T.ceil(T.log2(T.cast(i + 1024 + 1, ty="float64"))), ty="int32"
            )
            if x == 11:
                A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            x: T.let[T.int32] = T.Cast(
                "int32",
                T.ceil(T.log2(T.Cast("float64", i + 1025))),
            )
            A[i] = T.float32(0)

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_left_shift_lower_bound():
    """Integer bounds are propagated through left shift

    min(1 << i) = 1 << min(i)
                = 1 << 0
                = 1
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(1, i) >= 1:
                A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_left_shift_upper_bound():
    """Integer bounds are propagated through left shift

    max(31 << i) = 31 << max(i)
                 = 31 << 15
                 = 1015808
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(31, i) <= 1015808:
                A[i] = 0.0

    @T.function(private=True)
    def expected(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_left_shift_of_negative_value():
    """No const int bounds of left shift of negative value.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if -64 <= T.shift_left(-i, 4):
                A[i] = 0.0

    expected = before

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_left_shift_by_negative_value():
    """No const int bounds of left shift by negative bit count.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(16, -i) <= 16:
                A[i] = 0.0

    expected = before

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_conditional_floor_mod():
    """A regression test for negative floormod denominator

    Previously, simplifying this function could throw an error.  First, the
    `canonical_simplify` would rewrite `floormod(0-i,2)` to the equivalent
    `floormod(i,-2)`.  Then, the rewrite_simplifier would enter a
    constrained context in which `floormod(i,-2)==1`.  Passing this
    expression to `ModularSet::EnterConstraint`, which previously did not
    support a negative value for the second argument, threw an error.

    The analogous failure mode never occurred for `truncmod`, because
    `truncmod(0-i,2)` would be canonicalized to `truncmod(i, -2) * -1`, and
    the pattern matching in `ModularSet` didn't recognize the constant
    factor.

    This failure mode was resolved by supporting negative arguments in
    `ModularSet`, using the same sign convention as is used by
    `canonical_simplify`.
    """

    @T.function(private=True)
    def before(A: T.Tensor(1, "bool"), i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0

    @T.function(private=True)
    def expected(A: T.Tensor(1, "bool"), i: T.int32):
        if T.floormod(i, -2) == 0:
            A[0] = True

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_provable_condition_with_offset():
    """Use scoped-constraint to prove inequalities"""

    @T.function(private=True)
    def before(A: T.Tensor(1, "bool"), i: T.int32, j: T.int32):
        if i < j:
            A[0] = i < j + 1

    @T.function(private=True)
    def expected(A: T.Tensor(1, "bool"), i: T.int32, j: T.int32):
        if i < j:
            A[0] = True

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_most_restrictive_conditional():
    """Preferentially prove part of a compound conditional.

    Even if we cannot prove a conditional as true or false on its own,
    proving that a conditional must satisfy a stronger condition may
    allow for later rewrites.  For example, if it is known that `a <= b`,
    then `a >= b` cannot be proven, but can be reduced to `a == b`.
    """
    i, j, k = [T.dynamic(name, "int32") for name in "ijk"]
    tir_int = T.int32(0)

    test_cases = [
        (i <= tir_int, tir_int <= i, i == tir_int),
        (i <= tir_int, i != tir_int, i < tir_int),
        (i != tir_int, i <= tir_int, i < tir_int),
        (i != tir_int, tir_int <= i, tir_int < i),
        (i <= j, j <= i, j == i),
        (i <= j, i != j, i < j),
        (i != j, i <= j, i < j),
        (i != j, j <= i, j < i),
    ]

    for priors, expr_before, expr_after in test_cases:
        # well formed checker complains of undefined variables in condition
        @T.function(private=True, check_well_formed=False)
        def before_func(A: T.Tensor(1, "bool")):
            if priors:
                A[0] = expr_before

        # well formed checker complains of undefined variables in condition
        @T.function(private=True, check_well_formed=False)
        def expected_func(A: T.Tensor(1, "bool")):
            if priors:
                A[0] = expr_after

        after = _apply_simplify(before_func)
        tvm_tirx.ir.assert_structural_equal(after, expected_func)


def test_simplify_trivial_let_tensor_var():
    """A Bind used in a tensor definition should be retained"""

    @T.function(private=True)
    def before(A_ptr: T.handle("float32")):
        A_ptr_redef: T.let[T.handle("float32")] = A_ptr
        A = T.decl_tensor(A_ptr_redef, ty_args=[T.Tensor(1, "float32", data=A_ptr_redef)])
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_elem_offset():
    """A Bind used in a tensor definition should be retained"""

    @T.function(private=True)
    def before(A_ptr: T.handle("float32"), A_offset: T.int32):
        A_offset_redef = A_offset
        A = T.decl_tensor(
            A_ptr, ty_args=[T.Tensor(1, "float32", elem_offset=A_offset_redef, data=A_ptr)]
        )
        A[0] = 42.0

    @T.function(private=True)
    def expected(A_ptr: T.handle("float32"), A_offset: T.int32):
        A_offset_redef = A_offset
        A = T.decl_tensor(
            A_ptr, ty_args=[T.Tensor(1, "float32", elem_offset=A_offset_redef, data=A_ptr)]
        )
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_shape():
    """A Bind used in a tensor definition should be retained"""

    @T.function(private=True)
    def before(A_ptr: T.handle("float32"), A_size: T.int32):
        A_size_redef = A_size
        A = T.decl_tensor(A_ptr, ty_args=[T.Tensor([A_size_redef], "float32", data=A_ptr)])
        A[0] = 42.0

    @T.function(private=True)
    def expected(A_ptr: T.handle("float32"), A_size: T.int32):
        A_size_redef = A_size
        A = T.decl_tensor(A_ptr, ty_args=[T.Tensor([A_size_redef], "float32", data=A_ptr)])
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_stride():
    """A Bind used in a tensor definition should be retained"""

    @T.function(private=True)
    def before(A_ptr: T.handle("float32"), A_stride: T.int32):
        A_stride_redef = A_stride
        A = T.decl_tensor(
            A_ptr, ty_args=[T.Tensor(1, "float32", strides=[A_stride_redef], data=A_ptr)]
        )
        A[0] = 42.0

    @T.function(private=True)
    def expected(A_ptr: T.handle("float32"), A_stride: T.int32):
        A_stride_redef = A_stride
        A = T.decl_tensor(
            A_ptr, ty_args=[T.Tensor(1, "float32", strides=[A_stride_redef], data=A_ptr)]
        )
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_simplify_tensor_identity_well_formed():
    """Regression: Simplify must not diverge tensor identity between DeclTensor and TensorLoad.

    The simplifier's Dispatch calls analyzer_->Simplify() directly, bypassing
    normal ExprMutator dispatch.  If VisitTensorDef remaps a tensor at a DeclTensor
    site (e.g. inlining n_val -> n in the shape), TensorLoad inside a TensorStore
    value would NOT pick up the remap because VisitTensorUse is never called.
    This causes DeclTensor/TensorLoad tensor identity divergence.
    """

    @T.function(private=True)
    def before(A_ptr: T.handle("float32"), B_ptr: T.handle("float32"), n: T.int32):
        n_val = n
        A = T.decl_tensor(A_ptr, ty_args=[T.Tensor([n_val], "float32", data=A_ptr)])
        B = T.decl_tensor(B_ptr, ty_args=[T.Tensor([n_val], "float32", data=B_ptr)])
        B[0] = A[0]

    @T.function(private=True)
    def expected(A_ptr: T.handle("float32"), B_ptr: T.handle("float32"), n: T.int32):
        n_val = n
        A = T.decl_tensor(A_ptr, ty_args=[T.Tensor([n], "float32", data=A_ptr)])
        B = T.decl_tensor(B_ptr, ty_args=[T.Tensor([n], "float32", data=B_ptr)])
        B[0] = A[0]

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)
    tvm_tirx.tirx.analysis.verify_well_formed(after)


def test_tensor_shape_constraint():
    n = T.dynamic("n")

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function
        def main(A: T.Tensor((n * 32,), "float32")):
            A[T.min(T.int64(0), n)] = T.float32(0)

    n = T.dynamic("n")

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.function
        def main(A: T.Tensor((n * 32,), "float32")):
            A[T.int64(0)] = T.float32(0)

    after = tvm_tirx.tirx.transform.StmtSimplify()(Before)
    tvm_tirx.ir.assert_structural_equal(after["main"], Expected["main"])


def test_tensor_shape_constraint_with_offset():
    n = T.dynamic("n")

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function
        def main(A: T.Tensor((n * 32 + 1 - 2,), "float32")):
            A[T.min(T.int64(1), n)] = T.float32(0)

    n = T.dynamic("n")

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.function
        def main(A: T.Tensor((n * 32 + 1 - 2,), "float32")):
            A[T.int64(1)] = T.float32(0)

    after = tvm_tirx.tirx.transform.StmtSimplify()(Before)
    tvm_tirx.ir.assert_structural_equal(after["main"], Expected["main"])


def test_nested_if_elimination():
    @T.function(private=True)
    def before(a: T.Tensor((2, 8), "int32"), b: T.Tensor((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(
                i0 == 1 and 6 <= j0, 0, T.max(0, T.if_then_else(i0 == 1 and 6 <= j0, 0, a[i0, j0]))
            )

    @T.function(private=True)
    def expected(a: T.Tensor((2, 8), "int32"), b: T.Tensor((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(i0 == 1 and 6 <= j0, 0, T.max(0, a[i0, j0]))

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize("else_branch", [False, True])
@pytest.mark.parametrize("write_before_loop", [False, True])
def test_mutable_branch_predicate_preserves_while_bound(else_branch, write_before_loop):
    @T.function(private=True)
    def before(x: T.Tensor((1,), "int32"), count: T.Tensor((1,), "int32")):
        if T.constexpr(else_branch):
            if 8 <= x[0]:
                T.evaluate(0)
            else:
                if T.constexpr(write_before_loop):
                    x[0] = x[0] + 1
                while x[0] < 8 and count[0] == 0:
                    x[0] = x[0] + 1
        else:
            if x[0] < 8:
                if T.constexpr(write_before_loop):
                    x[0] = x[0] + 1
                while x[0] < 8 and count[0] == 0:
                    x[0] = x[0] + 1

    after = _apply_simplify(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_mutable_branch_predicate_preserves_later_load():
    @T.function(private=True)
    def before(x: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        if x[0] < 8:
            x[0] = x[0] + 1
            out[0] = T.Select(x[0] < 8, 1, 0)

    tvm_tirx.ir.assert_structural_equal(_apply_simplify(before), before)


def test_mutable_assert_does_not_constrain_later_load():
    @T.function(private=True)
    def before(x: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        assert x[0] < 8, "initial bound"
        x[0] = x[0] + 1
        out[0] = T.Select(x[0] < 8, 1, 0)

    tvm_tirx.ir.assert_structural_equal(_apply_simplify(before), before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
