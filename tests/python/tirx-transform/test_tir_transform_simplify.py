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
from tvm.script import ir as I
from tvm.script import tirx as T


def test_stmt_simplify():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_buffer((10,), "float32", data=A)
        C_ptr = T.decl_buffer((10,), "float32", data=C)
        n_val: T.let[T.int32] = 10
        for i in T.serial(n_val):
            if i < 12:
                A_ptr[i] = C_ptr[i]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StmtSimplify()(mod)["main"].body
    # Navigate through DeclBuffer nodes to reach the inner body
    while isinstance(body, tvm.tirx.DeclBuffer):
        body = body.body
    # After simplification, Bind is kept (not inlined) but the if is eliminated
    # since i < 12 is always true for i in 0..10.
    # Body is SeqStmt(Bind(n_val, 10), For(i, ...))
    stmts = body if isinstance(body, tvm.tirx.SeqStmt) else [body]
    # Find the For loop in the sequence
    for_stmt = [s for s in stmts if isinstance(s, tvm.tirx.For)]
    assert len(for_stmt) == 1, f"Expected one For loop, got {len(for_stmt)}"
    assert isinstance(for_stmt[0].body, tvm.tirx.BufferStore)


def test_thread_extent_simplify():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_buffer((10,), "float32", data=A)
        C_ptr = T.decl_buffer((10,), "float32", data=C)
        n_val: T.let[T.int32] = 10
        for tx in T.thread_binding(n_val, thread="threadIdx.x"):
            for ty in T.thread_binding(1, thread="threadIdx.y"):
                if tx + ty < 12:
                    A_ptr[tx] = C_ptr[tx + ty]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StmtSimplify()(mod)["main"].body
    # Navigate through DeclBuffer nodes to reach the inner body
    while isinstance(body, tvm.tirx.DeclBuffer):
        body = body.body
    # After simplification: Bind is kept but the if is eliminated
    # since tx + ty < 12 is always true for tx in 0..10 and ty = 0.
    stmts = list(body) if isinstance(body, tvm.tirx.SeqStmt) else [body]
    for_stmts = [s for s in stmts if isinstance(s, tvm.tirx.For)]
    assert len(for_stmts) >= 1, f"Expected For loop, got stmts: {[type(s).__name__ for s in stmts]}"
    # The outermost For is the tx loop
    tx_loop = for_stmts[0]
    assert isinstance(tx_loop, tvm.tirx.For)  # tx loop
    assert isinstance(tx_loop.body, tvm.tirx.For)  # ty loop
    assert isinstance(tx_loop.body.body, tvm.tirx.BufferStore)  # The if was eliminated


def test_if_likely():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.decl_buffer((32,), "float32", data=A)
        C_ptr = T.decl_buffer((1024,), "float32", data=C)
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            for ty in T.thread_binding(32, thread="threadIdx.y"):
                if T.likely(tx * 32 + ty < n):
                    if T.likely(tx * 32 + ty < n):
                        A_ptr[tx] = C_ptr[tx * 32 + ty]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StmtSimplify()(mod)["main"].body
    # With flat semantics, skip DeclBuffer/AllocBuffer siblings to find the For
    if isinstance(body, tvm.tirx.SeqStmt):
        for_stmts = [s for s in body.seq if isinstance(s, tvm.tirx.For)]
        body = for_stmts[0] if for_stmts else body
    # Structure: For(tx) -> For(ty) -> IfThenElse
    assert isinstance(body.body.body, tvm.tirx.IfThenElse)
    assert not isinstance(body.body.body.then_case, tvm.tirx.IfThenElse)


def test_loop_body_knows_dynamic_extent_is_positive():
    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((1,), "float32"), m: T.int32, n: T.int32):
        for i in T.serial(m, n // 4):
            if n // 4 - m > 0:
                A[0] = 1.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((1,), "float32"), m: T.int32, n: T.int32):
        for i in T.serial(m, n // 4):
            A[0] = 1.0

    after = tvm.tirx.transform.StmtSimplify()(tvm.IRModule.from_expr(before))["main"]
    tvm.ir.assert_structural_equal(after, expected)


def test_loop_var_does_not_escape_compacted_buffer_extent():
    @T.prim_func(private=True, s_tir=True)
    def before(a: T.handle):
        n = T.int64()
        A = T.match_buffer(a, (n,), "int32")
        tmp = T.alloc_buffer((n,), "int32")
        for i in range(n):
            length: T.let[T.int64] = T.ceildiv(n, T.shift_left(T.int64(1), i + 1))
            for j in range(length):
                tmp[j] = A[j]

    mod = tvm.s_tir.transform.CompactBufferAllocation()(tvm.IRModule.from_expr(before))
    after = tvm.tirx.transform.StmtSimplify()(mod)
    tvm.tirx.analysis.verify_well_formed(after)


def _apply_simplify(
    func,
    transitively_prove_inequalities=False,
    convert_boolean_to_and_of_ors=False,
    apply_constraints_to_boolean_branches=False,
):
    """Helper to apply simplify transform with config options."""
    config = {
        "tirx.StmtSimplify": {
            "transitively_prove_inequalities": transitively_prove_inequalities,
            "convert_boolean_to_and_of_ors": convert_boolean_to_and_of_ors,
            "apply_constraints_to_boolean_branches": apply_constraints_to_boolean_branches,
        }
    }
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(config=config):
        mod = tvm.tirx.transform.StmtSimplify()(mod)
    return mod["main"]


def test_load_store_noop():
    """Store of a value that was just read from the same location is a no-op."""

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0]

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_load_store_noop_after_simplify():
    """As test_load_store_noop, but requiring simplification to identify.

    Previously, a bug caused the self-assignment of a buffer to
    checked based on the pre-simplification assignment, not the
    post-simplification.  This test is to identify any similar
    regression.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0] + (5.0 - 5.0)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_nested_condition():
    """Nested IfThenElse with the same condition can be simplified.

    Requires const_int_bound to narrow scope of i within the
    conditional, or for rewrite_simplify to recognize the literal
    constraint.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_nested_provable_condition():
    """Simplify inner conditional using constraint from outer.

    Requires const_int_bound to narrow scope of i within the
    conditional.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i < 7:
                    A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_nested_var_condition():
    """Simplify inner conditional using constraint from outer.

    Requires for rewrite_simplify to recognize the repeated
    constraint.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i == n:
                    A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_altered_buffer_contents():
    """No simplification of data-dependent conditionals.

    A literal constraint must not be propagated if the values
    referenced may change.  TIR requires single assignment of
    variables, so Var objects may be assumed constant, but BufferLoad
    may not.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_negation_of_condition():
    """Use negation of outer condition to simplify innner.

    Within the body of an if statement, the negation of the
    condition is known to be false.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                if i != 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 1

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_negation_of_not_equal():
    """As test_negation_of_var_condition, but with a != outer condition.

    Because ConstIntBoundAnalyzer only tracks the min and max allowed
    values, the outer i!=5 condition does provide a constraint on the
    bounds.  This test relies on RewriteSimplifier to recognize
    ``i==5`` as the negation of a literal constraint.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                if i == 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                A[i] = 1

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_negation_of_var_condition():
    """As test_negation_of_var_condition, but with a dynamic condition.

    This simplification cannot be done with ConstIntBoundAnalyzer, and
    must rely on RewriteSimplifier recognizing the repeated literal.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i != n:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 1

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_literal_constraint_split_boolean_and():
    """Split a boolean AND into independent constraints

    A single if condition may impose multiple literal constraints.
    Each constraint that is ANDed together to form the condition
    should be treated as an independent constraint.  The use of n in
    the condition is to ensure we exercise RewriteSimplifier.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                if i == n:
                    A[i, j] = 0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                A[i, j] = 0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_literal_constraint_split_boolean_or():
    """Split a boolean OR into independent constraints

    Similar to test_literal_constraint_split_boolean_and, but splitting a
    boolean OR into independent conditions.  This uses the
    simplification that ``!(x || y) == !x && !y``.

    The use of ``n`` in the condition is to ensure we exercise
    RewriteSimplifier.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                A[i, j] = 2

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_prove_condition_using_let():
    """Simplify conditions using non-inlined let bindings

    Not all let bindings are inlined when they occur in later
    expressions.  However, even if they are not inlined, they may be
    used to prove the value of a condition.

    With flat Bind, the analyzer binds the variable to its value for
    constraint proving, which also substitutes the variable in later
    expressions.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if condition or i >= 3:
                A[i] = condition

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3  # noqa: F841
            A[i] = i < 3

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_prove_let_condition():
    """Simplify conditions using non-inlined let bindings

    With flat Bind, analyzer binds variable to value, which also
    substitutes the variable in later expressions.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if i < 3:
                if condition:
                    A[i] = condition

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3  # noqa: F841
            if i < 3:
                A[i] = T.bool(True)

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_prove_repeated_let_condition():
    """Simplify conditions using non-inlined let bindings

    With analyzer Bind, the variable is substituted with its value,
    so `if condition` becomes `if i < 3`, and within that context
    the inner `if condition` simplifies to True and is eliminated.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3
            if condition:
                if condition:
                    A[i] = condition

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition: T.let[T.bool] = i < 3  # noqa: F841
            if i < 3:
                A[i] = T.bool(True)

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_if_then_else_expr():
    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0, dtype="float32")

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_ceil_log2_int():
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = T.cast(
            T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32"
        )

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 4

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_ceil_log2_lower_bound():
    """Integer bounds are propagated through topi.math.ceil_log2

    With flat Bind, the Bind is kept even when the variable is unused
    after simplification. The if condition is still eliminated.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            x: T.let[T.int32] = T.cast(
                T.ceil(T.log2(T.cast(i + 1024 + 1, "float64"), dtype="float64"), dtype="float64"),
                dtype="int32",
            )
            if x == 11:
                A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            x: T.let[T.int32] = T.Cast(  # noqa: F841
                "int32",
                T.ceil(T.log2(T.Cast("float64", i + 1025))),
            )
            A[i] = T.float32(0)

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_shift_lower_bound():
    """Integer bounds are propagated through left shift

    min(1 << i) = 1 << min(i)
                = 1 << 0
                = 1
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(1, i, dtype="int32") >= 1:
                A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_shift_upper_bound():
    """Integer bounds are propagated through left shift

    max(31 << i) = 31 << max(i)
                 = 31 << 15
                 = 1015808
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(31, i, dtype="int32") <= 1015808:
                A[i] = 0.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_shift_of_negative_value():
    """No const int bounds of left shift of negative value.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if -64 <= T.shift_left(-i, 4, dtype="int32"):
                A[i] = 0.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_shift_by_negative_value():
    """No const int bounds of left shift by negative bit count.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(16, -i, dtype="int32") <= 16:
                A[i] = 0.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_remove_transitively_provable_condition():
    """Remove comparisons that may be proven using multiple others

    For example, the `0 < i` and `i <= j` conditions can be used to prove
    that `0 < j`.
    """
    i, j, k = [tvm.tirx.Var(name, "int32") for name in "ijk"]
    zero = tvm.tirx.IntImm("int32", 0)

    test_cases = [
        (tvm.tirx.all(zero < i, i <= j), zero < j, True),
        # Transitive comparisons from LT
        (tvm.tirx.all(i < j, j < k), i < k, True),
        (tvm.tirx.all(i < j, j == k), i < k, True),
        (tvm.tirx.all(i < j, j <= k), i < k, True),
        (tvm.tirx.all(i < j, j > k), i < k, False),
        (tvm.tirx.all(i < j, j >= k), i < k, False),
        (tvm.tirx.all(i < j, j != k), i < k, False),
        # Transitive comparisons from LE
        (tvm.tirx.all(i <= j, j < k), i < k, True),
        (tvm.tirx.all(i <= j, j == k), i == k, False),
        (tvm.tirx.all(i <= j, j == k), i <= k, True),
        (tvm.tirx.all(i <= j, j <= k), i <= k, True),
        (tvm.tirx.all(i <= j, j <= k), i < k, False),
        (tvm.tirx.all(i <= j, j > k), i < k, False),
        (tvm.tirx.all(i <= j, j >= k), i < k, False),
        (tvm.tirx.all(i <= j, j != k), i < k, False),
        # Transitive comparisons from GT
        (tvm.tirx.all(i > j, j > k), i > k, True),
        (tvm.tirx.all(i > j, j == k), i > k, True),
        (tvm.tirx.all(i > j, j >= k), i > k, True),
        (tvm.tirx.all(i > j, j < k), i > k, False),
        (tvm.tirx.all(i > j, j <= k), i > k, False),
        (tvm.tirx.all(i > j, j != k), i > k, False),
        # Transitive comparisons from GE
        (tvm.tirx.all(i >= j, j > k), i > k, True),
        (tvm.tirx.all(i >= j, j == k), i == k, False),
        (tvm.tirx.all(i >= j, j == k), i >= k, True),
        (tvm.tirx.all(i >= j, j >= k), i >= k, True),
        (tvm.tirx.all(i >= j, j >= k), i > k, False),
        (tvm.tirx.all(i >= j, j < k), i > k, False),
        (tvm.tirx.all(i >= j, j <= k), i > k, False),
        (tvm.tirx.all(i >= j, j != k), i > k, False),
        # GT or LT may be used to prove NE
        (tvm.tirx.all(i == j, j != k), i != k, True),
        (tvm.tirx.all(i == j, j < k), i != k, True),
        (tvm.tirx.all(i == j, j > k), i != k, True),
        (tvm.tirx.all(i == j, j != k), i < k, False),
        (tvm.tirx.all(i == j, j != k), i > k, False),
        # Because these are integers, x<y is equivalent to x <= y-1,
        # and may be used in equivalent simplifications.
        (tvm.tirx.all(i <= j - 1, j < k), i < k, True),
        (tvm.tirx.all(i <= j - 1, j == k), i < k, True),
        (tvm.tirx.all(i <= j - 1, j <= k), i < k, True),
        (tvm.tirx.all(i <= j - 1, j > k), i < k, False),
        (tvm.tirx.all(i <= j - 1, j >= k), i < k, False),
        (tvm.tirx.all(i <= j - 1, j != k), i < k, False),
        # Either or both inequalities may have an additive offset.
        (tvm.tirx.all(i <= j + 5, j <= k + 7), i <= k + 12, True),
        (tvm.tirx.all(i <= j + 5, j <= k + 7), i <= k + 11, False),
        # For floats, x < y + c1 and y < z + c2 implies that x < z + (c1 + c2).
        # Because this simplification applies to integers, transitive
        # application of LT or GT can give a tighter constraint.
        #
        # i < j + c1, j < k + c2
        # i <= j + c1 - 1, j <= k + c2 - 1
        # i + 1 - c1 <= j, j <= k + c2 - 1
        # i + 1 - c1 <= k + c2 - 1
        # i <= k + c1 + c2 - 2
        # i < k + (c1 + c2 - 1)
        #
        (tvm.tirx.all(i < j + 5, j < k + 7), i < k + 11, True),
        (tvm.tirx.all(i < j + 5, j < k + 7), i < k + 10, False),
    ]

    analyzer = tvm.arith.Analyzer()

    for priors, postulate, provable in test_cases:
        # well formed checker complains of undefined variables in condition
        @T.prim_func(private=True, check_well_formed=False, s_tir=True)
        def before_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = postulate

        priors_simplified = analyzer.canonical_simplify(priors)

        if provable:
            # well formed checker complains of undefined variables in condition
            @T.prim_func(private=True, check_well_formed=False, s_tir=True)
            def expected_func(A: T.Buffer(1, "bool")):
                if priors_simplified:
                    A[0] = True

        else:
            postulate_simplified = analyzer.canonical_simplify(postulate)

            # well formed checker complains of undefined variables in condition
            @T.prim_func(private=True, check_well_formed=False, s_tir=True)
            def expected_func(A: T.Buffer(1, "bool")):
                if priors_simplified:
                    A[0] = postulate_simplified

        after = _apply_simplify(before_func, transitively_prove_inequalities=True)
        tvm.ir.assert_structural_equal(after, expected_func)


def test_suppress_transitively_provable_condition():
    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        if i < j and j < k:
            A[0] = i < k

    expected = before

    after = _apply_simplify(before, transitively_prove_inequalities=False)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_ors():
    """If enabled, rewrite boolean expressions into AND of OR"""

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(3, "bool")):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_rewrite_as_and_of_ors():
    """Only rewrite into AND of OR when allowed"""

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    expected = before

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=False)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_ors_with_top_level_and():
    """The expression being rewritten may start with an AND

    Like test_rewrite_as_and_of_ors, but with an AndNode as the outermost
    booelan operator.  Even though it is primarily OR nodes that are
    being rewritten, the call to SimplifyAsAndOfOrs should apply to
    the outermost AndNode or OrNode in order to enable better
    simplification.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(4, "bool")):
        T.evaluate((A[0] or A[1]) and (A[1] or (A[0] and A[2] and A[3])))

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(4, "bool")):
        # If the simplification is applied to the OrNode, then a
        # redundant `(A[1] or A[0])` would't be canceled out.  When
        # applying SimplifyAsAndOfOrs to the top-level AndNode, the
        # internal representation is `[[0,1], [1,0], [1,2], [1,3]]`, and
        # the redundant `[1,0]` can be removed.
        #
        # If the simplification were only applied when encountering an
        # OrNode, the internal representation would be `[[0,1]]` during
        # the first call and `[[1,0], [1,2], [1,3]]` during the second
        # call.  As a result, the `[0,1]` and `[1,0]` representations
        # wouldn't occur within the same call, and the redundant `[1,0]`
        # wouldn't be removed.
        T.evaluate((A[0] or A[1]) and (A[1] or A[2]) and (A[1] or A[3]))

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_ors_with_simplification_between_groups():
    """Apply rewrite rules between OR groups that differ by a single element

    The expression `(k==20 and k!=30)` could be rewritten into `(k==20)`.
    However, by default these two terms must appear as part of an explicit part
    of the simplified expression.  The AndOfOr simplification checks for
    rewrite patterns of the form `(A or B) and (A or C)`, where `(B and C)` can
    simplify to a single expression `D`.  These can be rewritten to `(A or D)`.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (i == 0 or j == 10 or k != 30)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = i == 0 or j == 10 or k == 20

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_ors_with_simplification_between_reordered_groups():
    """Rewrite rules between OR groups do not depend on order

    Like test_rewrite_as_and_of_ors_with_simplification_between_groups, but the groups
    are ordered differently.  If this removes a group entirely, the result is
    ordered according to the first group in the expression.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (j == 10 or k != 30 or i == 0)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = j == 10 or k == 20 or i == 0

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_or_using_simplification_across_and():
    """Apply AndNode rewrites to non-adjacent expressions

    The RewriteSimplifier rules only check for simplifications between
    left/right branches of an And/Or node.  Simplifications that would require
    rearranging components in a chain of And/Or nodes are not performed.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (k == 20) and ((i == 0 or j == 10) and (k != 30))

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10) and (k == 20)

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_or_using_simplification_within_or():
    """Rewrite rules between OR groups do not depend on order

    The RewriteSimplifier rules only check for simplifications between
    left/right branches of an And/Or node.  Simplifications that would require
    rearranging components in a chain of And/Or nodes are not performed.

    This test validates that `(i == 20) or (i != 30)` can be rewritten to
    `(i != 30)`, even when there's an intervening clause between the
    clauses being simplified.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 20) or (j == 0) or (i != 30)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (j == 0) or (i != 30)

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


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

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(i, -2) == 0:
            A[0] = True

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_rhs_of_boolean_and_using_lhs():
    """Boolean expressions can introduce contexts.

    In `A and B`, the result of `B` only matters when `A` is
    true, and can be simplified under that context.  This test
    simplifies `n < 10` under the assumption that `n < 5`.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 and n < 10

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_and_using_rhs():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_rhs_of_boolean_and_using_lhs, but using the RHS to
    simplify the LHS.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 and n < 5

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_rhs_of_boolean_or_using_lhs():
    """Boolean expressions can introduce contexts.

    In `A or B`, the result of `B` only matters when `A` is false, so
    `B` can be simplified under the assumption that `A` is false.
    This test simplifies `n < 5` under the assumption that `!(n < 10)`
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 or n < 5

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_or_using_rhs():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_rhs_of_boolean_or_using_lhs, but using the RHS to
    simplify the LHS.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 or n < 10

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_rhs_of_boolean_and_using_lhs_without_const():
    """Boolean expressions can introduce contexts.

    Like test_simplify_rhs_of_boolean_and_using_lhs, but with variables in
    the conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5 and n < m + 10

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5

    after = _apply_simplify(
        before, apply_constraints_to_boolean_branches=True, transitively_prove_inequalities=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_and_using_rhs_without_const():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_lhs_of_boolean_and_using_rhs, but with variables in
    the conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10 and n < m + 5

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5

    after = _apply_simplify(
        before, apply_constraints_to_boolean_branches=True, transitively_prove_inequalities=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_rhs_of_boolean_or_using_lhs_without_const():
    """Boolean expressions can introduce contexts.

    Like test_simplify_rhs_of_boolean_or_using_lhs, but with variables in the
    conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10 or n < m + 5

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10

    after = _apply_simplify(
        before, apply_constraints_to_boolean_branches=True, transitively_prove_inequalities=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_or_using_rhs_without_const():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_lhs_of_boolean_or_using_rhs, but with variables in the
    conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5 or n < m + 10

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10

    after = _apply_simplify(
        before, apply_constraints_to_boolean_branches=True, transitively_prove_inequalities=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_provable_condition_with_offset():
    """Use scoped-constraint to prove inequalities"""

    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32):
        if i < j:
            A[0] = i < j + 1

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32):
        if i < j:
            A[0] = True

    after = _apply_simplify(before, transitively_prove_inequalities=False)
    tvm.ir.assert_structural_equal(after, expected)


def test_most_restrictive_conditional():
    """Preferentially prove part of a compound conditional.

    Even if we cannot prove a conditional as true or false on its own,
    proving that a conditional must satisfy a stronger condition may
    allow for later rewrites.  For example, if it is known that `a <= b`,
    then `a >= b` cannot be proven, but can be reduced to `a == b`.
    """
    i, j, k = [tvm.tirx.Var(name, "int32") for name in "ijk"]
    tir_int = tvm.tirx.IntImm("int32", 0)

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
        @T.prim_func(private=True, check_well_formed=False, s_tir=True)
        def before_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = expr_before

        # well formed checker complains of undefined variables in condition
        @T.prim_func(private=True, check_well_formed=False, s_tir=True)
        def expected_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = expr_after

        after = _apply_simplify(before_func)
        tvm.ir.assert_structural_equal(after, expected_func)


def test_simplify_trivial_let_buffer_var():
    """A Bind used in a buffer definition should be retained"""

    @T.prim_func(private=True, s_tir=True)
    def before(A_ptr: T.handle("float32")):
        A_ptr_redef: T.let[T.handle("float32")] = A_ptr
        A = T.decl_buffer(1, "float32", data=A_ptr_redef)
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_elem_offset():
    """A Bind used in a buffer definition should be retained"""

    @T.prim_func(private=True, s_tir=True)
    def before(A_ptr: T.handle("float32"), A_offset: T.int32):
        A_offset_redef = A_offset
        A = T.decl_buffer(1, "float32", elem_offset=A_offset_redef, data=A_ptr)
        A[0] = 42.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A_ptr: T.handle("float32"), A_offset: T.int32):
        A_offset_redef = A_offset
        A = T.decl_buffer(1, "float32", elem_offset=A_offset_redef, data=A_ptr)
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_shape():
    """A Bind used in a buffer definition should be retained"""

    @T.prim_func(private=True, s_tir=True)
    def before(A_ptr: T.handle("float32"), A_size: T.int32):
        A_size_redef = A_size
        A = T.decl_buffer([A_size_redef], "float32", data=A_ptr)
        A[0] = 42.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A_ptr: T.handle("float32"), A_size: T.int32):
        A_size_redef = A_size
        A = T.decl_buffer([A_size_redef], "float32", data=A_ptr)
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_stride():
    """A Bind used in a buffer definition should be retained"""

    @T.prim_func(private=True, s_tir=True)
    def before(A_ptr: T.handle("float32"), A_stride: T.int32):
        A_stride_redef = A_stride
        A = T.decl_buffer(1, "float32", strides=[A_stride_redef], data=A_ptr)
        A[0] = 42.0

    @T.prim_func(private=True, s_tir=True)
    def expected(A_ptr: T.handle("float32"), A_stride: T.int32):
        A_stride_redef = A_stride
        A = T.decl_buffer(1, "float32", strides=[A_stride_redef], data=A_ptr)
        A[0] = 42.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_buffer_identity_well_formed():
    """Regression: Simplify must not diverge buffer identity between DeclBuffer and BufferLoad.

    The simplifier's VisitExpr calls analyzer_->Simplify() directly, bypassing
    normal ExprMutator dispatch.  If VisitBufferDef remaps a buffer at a DeclBuffer
    site (e.g. inlining n_val -> n in the shape), BufferLoad inside a BufferStore
    value would NOT pick up the remap because VisitBufferUse is never called.
    This causes DeclBuffer/BufferLoad buffer identity divergence.
    """

    @T.prim_func(private=True, s_tir=True)
    def before(A_ptr: T.handle("float32"), B_ptr: T.handle("float32"), n: T.int32):
        n_val = n
        A = T.decl_buffer([n_val], "float32", data=A_ptr)
        B = T.decl_buffer([n_val], "float32", data=B_ptr)
        B[0] = A[0]

    after = _apply_simplify(before)
    tvm.tirx.analysis.verify_well_formed(after)


def test_buffer_shape_constraint():
    @I.ir_module(check_well_formed=False)
    class Before:
        @T.prim_func(s_tir=True)
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32,), "float32")
            A[T.min(T.int64(0), n)] = T.float32(0)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.prim_func(s_tir=True)
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32,), "float32")
            A[T.int64(0)] = T.float32(0)

    after = tvm.tirx.transform.StmtSimplify()(Before)
    tvm.ir.assert_structural_equal(after["main"], Expected["main"])


def test_buffer_shape_constraint_with_offset():
    @I.ir_module(check_well_formed=False)
    class Before:
        @T.prim_func(s_tir=True)
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32 + 1 - 2,), "float32")
            A[T.min(T.int64(1), n)] = T.float32(0)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.prim_func(s_tir=True)
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32 + 1 - 2,), "float32")
            A[T.int64(1)] = T.float32(0)

    after = tvm.tirx.transform.StmtSimplify()(Before)
    tvm.ir.assert_structural_equal(after["main"], Expected["main"])


def test_nested_if_elimination():
    @T.prim_func(private=True, s_tir=True)
    def before(a: T.Buffer((2, 8), "int32"), b: T.Buffer((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(
                i0 == 1 and 6 <= j0, 0, T.max(0, T.if_then_else(i0 == 1 and 6 <= j0, 0, a[i0, j0]))
            )

    @T.prim_func(private=True, s_tir=True)
    def expected(a: T.Buffer((2, 8), "int32"), b: T.Buffer((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(i0 == 1 and 6 <= j0, 0, T.max(0, a[i0, j0]))

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
