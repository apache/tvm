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
from tvm.script import tir as T, ir as I


def test_stmt_simplify():
    @T.prim_func(private=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.Buffer((10,), "float32", data=A)
        C_ptr = T.Buffer((10,), "float32", data=C)
        n_val: T.int32 = 10
        for i in T.serial(n_val):
            if i < 12:
                A_ptr[i] = C_ptr[i]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    # After simplification, LetStmt -> For -> BufferStore (if is eliminated since i < 12 is always true for i in 0..10)
    assert isinstance(body.body, tvm.tir.BufferStore)


def test_thread_extent_simplify():
    @T.prim_func(private=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.Buffer((10,), "float32", data=A)
        C_ptr = T.Buffer((10,), "float32", data=C)
        n_val: T.int32 = 10
        for tx in T.thread_binding(n_val, thread="threadIdx.x"):
            for ty in T.thread_binding(1, thread="threadIdx.y"):
                if tx + ty < 12:
                    A_ptr[tx] = C_ptr[tx + ty]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    # After simplification: For(tx) -> For(ty) -> BufferStore
    # The LetStmt and if are eliminated since tx + ty < 12 is always true for tx in 0..10 and ty = 0
    assert isinstance(body, tvm.tir.For)  # tx loop
    assert isinstance(body.body, tvm.tir.For)  # ty loop
    assert isinstance(body.body.body, tvm.tir.BufferStore)  # The if was eliminated


def test_if_likely():
    @T.prim_func(private=True)
    def func(A: T.handle("float32"), C: T.handle("float32"), n: T.int32):
        A_ptr = T.Buffer((32,), "float32", data=A)
        C_ptr = T.Buffer((1024,), "float32", data=C)
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            for ty in T.thread_binding(32, thread="threadIdx.y"):
                if T.likely(tx * 32 + ty < n):
                    if T.likely(tx * 32 + ty < n):
                        A_ptr[tx] = C_ptr[tx * 32 + ty]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    # Structure: For(tx) -> For(ty) -> IfThenElse
    assert isinstance(body.body.body, tvm.tir.IfThenElse)
    assert not isinstance(body.body.body.then_case, tvm.tir.IfThenElse)


def _apply_simplify(
    func,
    transitively_prove_inequalities=False,
    convert_boolean_to_and_of_ors=False,
    apply_constraints_to_boolean_branches=False,
    propagate_knowns_to_prove_conditional=False,
    propagate_knowns_to_simplify_expressions=False,
):
    """Helper to apply simplify transform with config options."""
    config = {
        "tir.Simplify": {
            "transitively_prove_inequalities": transitively_prove_inequalities,
            "convert_boolean_to_and_of_ors": convert_boolean_to_and_of_ors,
            "apply_constraints_to_boolean_branches": apply_constraints_to_boolean_branches,
            "propagate_knowns_to_prove_conditional": propagate_knowns_to_prove_conditional,
            "propagate_knowns_to_simplify_expressions": propagate_knowns_to_simplify_expressions,
        }
    }
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(config=config):
        mod = tvm.tir.transform.Simplify()(mod)
    return mod["main"]


def test_load_store_noop():
    """Store of a value that was just read from the same location is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0]

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0] + (5.0 - 5.0)

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i < 7:
                    A[i] = 0.0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i == n:
                    A[i] = 0.0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                if i != 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                if i == 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i != n:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                if i == n:
                    A[i, j] = 0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2

    @T.prim_func(private=True)
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
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            if condition or i >= 3:
                A[i] = condition

    @T.prim_func(private=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            A[i] = condition

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_prove_let_condition():
    """Simplify conditions using non-inlined let bindings

    Not all let bindings are inlined when they occur in later
    expressions.  However, even if they are not inlined, they may be
    used to prove the value of a condition.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            if i < 3:
                if condition:
                    A[i] = condition

    @T.prim_func(private=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            if i < 3:
                A[i] = condition

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_prove_repeated_let_condition():
    """Simplify conditions using non-inlined let bindings

    A variable may be used as a literal constraint, and be recognized
    as being True within the context of the constraint.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                if condition:
                    A[i] = condition

    @T.prim_func(private=True)
    def expected(A: T.Buffer(4, "bool")):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                A[i] = True

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_if_then_else_expr():
    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0, dtype="float32")

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_ceil_log2_int():
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = T.cast(
            T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32"
        )

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 4

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_ceil_log2_lower_bound():
    """Integer bounds are propagated through topi.math.ceil_log2"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            x = T.cast(
                T.ceil(T.log2(T.cast(i + 1024 + 1, "float64"), dtype="float64"), dtype="float64"),
                dtype="int32",
            )
            if x == 11:
                A[i] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_left_shift_lower_bound():
    """Integer bounds are propagated through left shift

    min(1 << i) = 1 << min(i)
                = 1 << 0
                = 1
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(1, i, dtype="int32") >= 1:
                A[i] = 0.0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(31, i, dtype="int32") <= 1015808:
                A[i] = 0.0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
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
    i, j, k = [tvm.tir.Var(name, "int32") for name in "ijk"]
    zero = tvm.tir.IntImm("int32", 0)

    test_cases = [
        (tvm.tir.all(zero < i, i <= j), zero < j, True),
        # Transitive comparisons from LT
        (tvm.tir.all(i < j, j < k), i < k, True),
        (tvm.tir.all(i < j, j == k), i < k, True),
        (tvm.tir.all(i < j, j <= k), i < k, True),
        (tvm.tir.all(i < j, j > k), i < k, False),
        (tvm.tir.all(i < j, j >= k), i < k, False),
        (tvm.tir.all(i < j, j != k), i < k, False),
        # Transitive comparisons from LE
        (tvm.tir.all(i <= j, j < k), i < k, True),
        (tvm.tir.all(i <= j, j == k), i == k, False),
        (tvm.tir.all(i <= j, j == k), i <= k, True),
        (tvm.tir.all(i <= j, j <= k), i <= k, True),
        (tvm.tir.all(i <= j, j <= k), i < k, False),
        (tvm.tir.all(i <= j, j > k), i < k, False),
        (tvm.tir.all(i <= j, j >= k), i < k, False),
        (tvm.tir.all(i <= j, j != k), i < k, False),
        # Transitive comparisons from GT
        (tvm.tir.all(i > j, j > k), i > k, True),
        (tvm.tir.all(i > j, j == k), i > k, True),
        (tvm.tir.all(i > j, j >= k), i > k, True),
        (tvm.tir.all(i > j, j < k), i > k, False),
        (tvm.tir.all(i > j, j <= k), i > k, False),
        (tvm.tir.all(i > j, j != k), i > k, False),
        # Transitive comparisons from GE
        (tvm.tir.all(i >= j, j > k), i > k, True),
        (tvm.tir.all(i >= j, j == k), i == k, False),
        (tvm.tir.all(i >= j, j == k), i >= k, True),
        (tvm.tir.all(i >= j, j >= k), i >= k, True),
        (tvm.tir.all(i >= j, j >= k), i > k, False),
        (tvm.tir.all(i >= j, j < k), i > k, False),
        (tvm.tir.all(i >= j, j <= k), i > k, False),
        (tvm.tir.all(i >= j, j != k), i > k, False),
        # GT or LT may be used to prove NE
        (tvm.tir.all(i == j, j != k), i != k, True),
        (tvm.tir.all(i == j, j < k), i != k, True),
        (tvm.tir.all(i == j, j > k), i != k, True),
        (tvm.tir.all(i == j, j != k), i < k, False),
        (tvm.tir.all(i == j, j != k), i > k, False),
        # Because these are integers, x<y is equivalent to x <= y-1,
        # and may be used in equivalent simplifications.
        (tvm.tir.all(i <= j - 1, j < k), i < k, True),
        (tvm.tir.all(i <= j - 1, j == k), i < k, True),
        (tvm.tir.all(i <= j - 1, j <= k), i < k, True),
        (tvm.tir.all(i <= j - 1, j > k), i < k, False),
        (tvm.tir.all(i <= j - 1, j >= k), i < k, False),
        (tvm.tir.all(i <= j - 1, j != k), i < k, False),
        # Either or both inequalities may have an additive offset.
        (tvm.tir.all(i <= j + 5, j <= k + 7), i <= k + 12, True),
        (tvm.tir.all(i <= j + 5, j <= k + 7), i <= k + 11, False),
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
        (tvm.tir.all(i < j + 5, j < k + 7), i < k + 11, True),
        (tvm.tir.all(i < j + 5, j < k + 7), i < k + 10, False),
    ]

    analyzer = tvm.arith.Analyzer()

    for priors, postulate, provable in test_cases:
        # well formed checker complains of undefined variables in condition
        @T.prim_func(private=True, check_well_formed=False)
        def before_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = postulate

        priors_simplified = analyzer.canonical_simplify(priors)

        if provable:
            # well formed checker complains of undefined variables in condition
            @T.prim_func(private=True, check_well_formed=False)
            def expected_func(A: T.Buffer(1, "bool")):
                if priors_simplified:
                    A[0] = True

        else:
            postulate_simplified = analyzer.canonical_simplify(postulate)

            # well formed checker complains of undefined variables in condition
            @T.prim_func(private=True, check_well_formed=False)
            def expected_func(A: T.Buffer(1, "bool")):
                if priors_simplified:
                    A[0] = postulate_simplified

        after = _apply_simplify(before_func, transitively_prove_inequalities=True)
        tvm.ir.assert_structural_equal(after, expected_func)


def test_suppress_transitively_provable_condition():
    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        if i < j and j < k:
            A[0] = i < k

    expected = before

    after = _apply_simplify(before, transitively_prove_inequalities=False)
    tvm.ir.assert_structural_equal(after, expected)


def test_rewrite_as_and_of_ors():
    """If enabled, rewrite boolean expressions into AND of OR"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    @T.prim_func(private=True)
    def expected(A: T.Buffer(3, "bool")):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))

    after = _apply_simplify(before, convert_boolean_to_and_of_ors=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_rewrite_as_and_of_ors():
    """Only rewrite into AND of OR when allowed"""

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "bool")):
        T.evaluate((A[0] or A[1]) and (A[1] or (A[0] and A[2] and A[3])))

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (i == 0 or j == 10 or k != 30)

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (j == 10 or k != 30 or i == 0)

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (k == 20) and ((i == 0 or j == 10) and (k != 30))

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 20) or (j == 0) or (i != 30)

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 and n < 10

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_and_using_rhs():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_rhs_of_boolean_and_using_lhs, but using the RHS to
    simplify the LHS.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 and n < 5

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 or n < 5

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    after = _apply_simplify(before, apply_constraints_to_boolean_branches=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_lhs_of_boolean_or_using_rhs():
    """Boolean expressions can introduce contexts for their arguments.

    Like test_simplify_rhs_of_boolean_or_using_lhs, but using the RHS to
    simplify the LHS.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 or n < 10

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5 and n < m + 10

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10 and n < m + 5

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10 or n < m + 5

    @T.prim_func(private=True)
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

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 5 or n < m + 10

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "bool"), n: T.int32, m: T.int32):
        A[0] = n < m + 10

    after = _apply_simplify(
        before, apply_constraints_to_boolean_branches=True, transitively_prove_inequalities=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_provable_condition_with_offset():
    """Use scoped-constraint to prove inequalities"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "bool"), i: T.int32, j: T.int32):
        if i < j:
            A[0] = i < j + 1

    @T.prim_func(private=True)
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
    i, j, k = [tvm.tir.Var(name, "int32") for name in "ijk"]
    tir_int = tvm.tir.IntImm("int32", 0)

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
        @T.prim_func(private=True, check_well_formed=False)
        def before_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = expr_before

        # well formed checker complains of undefined variables in condition
        @T.prim_func(private=True, check_well_formed=False)
        def expected_func(A: T.Buffer(1, "bool")):
            if priors:
                A[0] = expr_after

        after = _apply_simplify(before_func)
        tvm.ir.assert_structural_equal(after, expected_func)


def test_altered_buffer_contents_with_propagation():
    """Propagation of data-dependent conditionals.

    A literal constraint must not be propagated if the values
    referenced may change.  TIR requires single assignment of
    variables, so Var objects may be assumed constant, but BufferLoad
    may not.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            # If the simplifier incorrectly uses the invalidated
            # A[0]==n condition required to reach this point, then it
            # will incorrectly simplify to the then-case.  If the
            # simplifier correctly determines that A[0] now contains
            # n+1, then it will correctly simplify to the else-case.
            if A[0] == n:
                A[0] = 5
            else:
                A[0] = 10

    @T.prim_func(private=True)
    def expected(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            A[0] = 10

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_possibly_altered_buffer_contents():
    """No simplification of data-dependent conditionals.

    Like test_altered_buffer_contents_with_propagation, but the `m==0` conditional
    prevents the value of `A[0]` from being known at the point of the
    inner conditional, either as `A[0] == n` from the outer
    conditional or as `A[0] == n+1` from the write statement.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer((1,), "int32"), n: T.int32, m: T.int32):
        if A[0] == n:
            if m == 0:
                A[0] = A[0] + 1

            if A[0] == n:
                A[0] = 5
            else:
                A[0] = 10

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_input_assumption():
    """A T.assume annotation may be used to simplify"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32"), n: T.int32):
        T.evaluate(T.assume(n == 0))
        if n == 0:
            A[0] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32"), n: T.int32):
        T.evaluate(T.assume(n == 0))
        A[0] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_simplify_from_scoped_input_assumption():
    """A T.assume inside a scope may not apply outside that scope"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32"), n: T.int32, m: T.int32):
        if m == 0:
            T.evaluate(T.assume(n == 0))

        if n == 0:
            A[0] = 42

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_conditional_using_buffer_value():
    """Simplify a conditional using the known value in the buffer"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0

        if A[0] == 0:
            A[0] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_keep_expression_simplify_using_buffer_value():
    """Do not simplify expressions in general using known values in the buffer

    For now, because this is equivalent to inlining, preventing this
    usage from occurring.  Known buffer values may be used to prove
    conditionals, but should not be used for other simplifications.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32"), B: T.Buffer(1, "int32")):
        A[0] = 0
        B[0] = A[0]

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_conditional_in_loop_using_buffer_value():
    """Simplify a conditional using the known value in the buffer

    Like test_simplify_conditional_using_buffer_value, but the value used
    to simplify is set in a previous loop.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

        for j in T.serial(16):
            if A[j] == j:
                B[j] = 42
            else:
                B[j] = 100

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

        for j in T.serial(16):
            B[j] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_buffer_assumption():
    """A T.assume may apply to a buffer's contents"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 0))

        if A[0] == 0:
            A[0] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 0))
        A[0] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_buffer_assumption_in_loop():
    """An assumption about buffer contents may apply to a range"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == i))

        for i in T.serial(16):
            if A[i] < 100:
                A[i] = 0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == i))

        for i in T.serial(16):
            A[i] = 0

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_partially_known_buffer_conditional():
    """An assumption about buffer contents may apply to only part of a buffer"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if 14 <= i:
                T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                if A[i] == 0:
                    A[i] = 42

            else:
                if A[i] == 0:
                    A[i] = 100

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if 14 <= i:
                T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                A[i] = 42

            else:
                if A[i] == 0:
                    A[i] = 100

    after = _apply_simplify(
        before,
        propagate_knowns_to_prove_conditional=True,
        apply_constraints_to_boolean_branches=True,
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_partially_known_buffer_expression():
    """An assumption about buffer contents may apply to only part of a buffer

    Like test_simplify_using_partially_known_buffer_conditional, but the
    conditional is expressed as part of T.assume, instead of in the
    control flow.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                if A[i] == 0:
                    A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                A[i] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_simplification_if_predicate_not_met():
    """Assumptions about buffer contents must apply to all cases to be used

    Like test_simplify_using_partial_buffer_assumption_in_loop, but the
    predicate in the second loop does not match the predicate in the
    first loop.  Therefore, the `T.assume` refers to a different set
    of indices.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if 14 <= i:
                T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if i < 14:
                if A[i] == 0:
                    A[i] = 42

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_simplify_using_invalidated_scoped_constraint():
    """A write may not be used for proofs outside its conditional"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i == 0:
                A[i] = 0

            if A[i] == 0:
                A[i] = 42

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_simplify_using_overwritten_value():
    """A write that may have been overwritten may not be treated as known

    The appearance of "A[i] = 5" must prevent the earlier constraint
    from being used for simplification.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_simplify_using_loop_dependent_buffer_value():
    """Do not simplify assuming reads are invariant

    If a buffer's value changes across loop iterations, the buffer's
    value before the loop should not be used to simplify conditionals
    within the loop.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32"), B: T.Buffer(1, "int32")):
        B[0] = 0
        for i in T.serial(16):
            if B[0] < 10:
                B[0] = A[i] * 2 + B[0]
            else:
                B[0] = A[i] + B[0]

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_prior_to_overwritten_value():
    """A known value may be used until it is overwritten

    Like test_no_simplify_using_overwritten_value, but the use of the
    known `A[i]` value occurs before it is overwritten.

    Like test_no_simplify_using_loop_dependent_buffer_value, but the loop
    iterations are all independent.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if A[i] == 0:
                A[i] = 17

            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            A[i] = 17

            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_element_wise_using_pre_loop_buffer_value():
    """Allow data-Do not simplify assuming reads are invariant

    If an element-wise loop reads and overwrites a buffer value, the
    pre-loop buffer value may be used to simplify conditions that
    occur prior to the write.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            if B[i] < 10:
                B[i] = A[i] * 2 + B[i]
            else:
                B[i] = A[i] + B[i]

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            B[i] = A[i] * 2 + B[i]

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_non_conditional():
    """Propagate a known value to later expressions."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = A[0] + 1

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = 1

    after = _apply_simplify(before, propagate_knowns_to_simplify_expressions=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_simplify_non_conditional():
    """Propagate a known value to later expressions.

    Like test_simplify_non_conditional, but with data-propagation turned off.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = A[0] + 1

    expected = before

    after = _apply_simplify(before, propagate_knowns_to_simplify_expressions=False)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_transitive_known_buffer_value():
    """Propagate known buffer values

    If a known value of a buffer depends on another known value, it
    can be tracked backwards through both.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 0))

        A[0] = A[0] + 1
        A[0] = A[0] + 1
        A[0] = A[0] + 1

        if A[0] == 3:
            A[0] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        T.evaluate(T.assume(A[0] == 0))

        A[0] = A[0] + 1
        A[0] = A[0] + 1
        A[0] = A[0] + 1

        A[0] = 42

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_ramp_index_broadcast_value():
    """Simplifications involving buffer loads with ramp indices"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "int32")):
        A[T.ramp(0, 1, 4)] = T.broadcast(0, 4)

        if A[0] == 0:
            A[0] = 42

        if A[1] == 0:
            A[1] = 60

    @T.prim_func(private=True)
    def expected(A: T.Buffer(4, "int32")):
        A[T.ramp(0, 1, 4)] = T.broadcast(0, 4)

        A[0] = 42
        A[1] = 60

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_ramp_index_ramp_value():
    """Simplifications involving buffer loads with ramp indices"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(4, "int32")):
        A[T.ramp(0, 1, 4)] = T.ramp(11, 1, 4)

        if A[0] == 11:
            A[0] = 42

        if A[1] == 12:
            A[1] = 60

    @T.prim_func(private=True)
    def expected(A: T.Buffer(4, "int32")):
        A[T.ramp(0, 1, 4)] = T.ramp(11, 1, 4)

        A[0] = 42
        A[1] = 60

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_partially_proven_buffer_value_gather():
    """Propagate known buffer values in part of buffer.

    Even if a constraint can't be solved for all values in an
    assignment, it may be provable in part of a buffer.  Here, the
    known 0 values in the padding of A produces known 0 values in the
    padding of B.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(24, "int32"), B: T.Buffer(24, "int32"), F: T.Buffer(3, "int32")):
        # A has non-zero values only in the range 3 <= i < 17
        for i in T.serial(24):
            T.evaluate(T.assume(((3 <= i) and (i < 17)) or A[i] == 0))

        # After convoluting with F, B has non-zero values only in the
        # range 3 <= i < 19.
        for i in T.serial(24):
            B[i] = 0
            for f in T.serial(3):
                if 0 <= i - f:
                    B[i] = B[i] + A[i - f] * F[f]

        # Which means that this loop is unnecessary.  It would be
        # removed entirely in tir.transform.RemoveNoOp, but here we
        # want to test that the simplification works as intended.
        for i in T.serial(24):
            if i < 3 or 19 <= i:
                if B[i] != 0:
                    B[i] = 0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(24, "int32"), B: T.Buffer(24, "int32"), F: T.Buffer(3, "int32")):
        for i in T.serial(24):
            T.evaluate(T.assume(((3 <= i) and (i < 17)) or A[i] == 0))

        for i in T.serial(24):
            B[i] = 0
            for f in T.serial(3):
                if 0 <= i - f:
                    B[i] = B[i] + A[i - f] * F[f]

        for i in T.serial(24):
            if i < 3 or 19 <= i:
                T.evaluate(0)

    after = _apply_simplify(
        before, transitively_prove_inequalities=True, propagate_knowns_to_prove_conditional=True
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_using_partially_proven_buffer_value_scatter():
    """Propagate known buffer values in part of buffer.

    Like test_simplify_using_partially_proven_buffer_value_gather, but the
    compute loop is over the input buffer A, rather than the output
    buffer B.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(24, "int32"), B: T.Buffer(24, "int32"), F: T.Buffer(3, "int32")):
        # A has non-zero values only in the range 3 <= i < 17
        for i in T.serial(24):
            T.evaluate(T.assume(((3 <= i) and (i < 17)) or A[i] == 0))

        for i in T.serial(24):
            B[i] = 0

        # After convoluting with F, B has non-zero values only in the
        # range 3 <= i < 19.
        for i in T.serial(24):
            for f in T.serial(3):
                if i + f >= 0 and i + f < 24:
                    B[i + f] = B[i + f] + A[i] * F[f]

        # Which means that this loop is unnecessary.  It actually gets
        # removed in tir.transform.RemoveNoOp, but here we want to
        # test that the simplification works as intended.
        for i in T.serial(24):
            if i < 3 or 19 <= i:
                if B[i] != 0:
                    B[i] = 0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(24, "int32"), B: T.Buffer(24, "int32"), F: T.Buffer(3, "int32")):
        for i in T.serial(24):
            T.evaluate(T.assume(((3 <= i) and (i < 17)) or A[i] == 0))

        for i in T.serial(24):
            B[i] = 0

        for i in T.serial(24):
            for f in T.serial(3):
                if i + f < 24:
                    B[i + f] = B[i + f] + A[i] * F[f]

        for i in T.serial(24):
            if i < 3 or 19 <= i:
                T.evaluate(0)

    after = _apply_simplify(before, propagate_knowns_to_prove_conditional=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_buffer_store():
    """Simplification using prior known"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = 5
        A[0] = A[0] + 7

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 5
        A[0] = 12

    after = _apply_simplify(before, propagate_knowns_to_simplify_expressions=True)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_buffer_var():
    """A LetStmt used in a buffer definition should be retained"""

    @T.prim_func(private=True)
    def before(A_ptr: T.handle("float32")):
        A_ptr_redef: T.handle("float32") = A_ptr
        A = T.decl_buffer(1, "float32", data=A_ptr_redef)
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_elem_offset():
    """A LetStmt used in a buffer definition should be retained"""

    @T.prim_func(private=True)
    def before(A_ptr: T.handle("float32"), A_offset: T.int32):
        A_offset_redef = A_offset
        A = T.decl_buffer(1, "float32", elem_offset=A_offset_redef, data=A_ptr)
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_shape():
    """A LetStmt used in a buffer definition should be retained"""

    @T.prim_func(private=True)
    def before(A_ptr: T.handle("float32"), A_size: T.int32):
        A_size_redef = A_size
        A = T.decl_buffer([A_size_redef], "float32", data=A_ptr)
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_simplify_trivial_let_stride():
    """A LetStmt used in a buffer definition should be retained"""

    @T.prim_func(private=True)
    def before(A_ptr: T.handle("float32"), A_stride: T.int32):
        A_stride_redef = A_stride
        A = T.decl_buffer(1, "float32", strides=[A_stride_redef], data=A_ptr)
        A[0] = 42.0

    expected = before

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_buffer_shape_constraint():
    @I.ir_module(check_well_formed=False)
    class Before:
        @T.prim_func
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32,), "float32")
            A[T.min(T.int64(0), n)] = T.float32(0)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.prim_func
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32,), "float32")
            A[T.int64(0)] = T.float32(0)

    after = tvm.tir.transform.Simplify()(Before)
    tvm.ir.assert_structural_equal(after["main"], Expected["main"])


def test_buffer_shape_constraint_with_offset():
    @I.ir_module(check_well_formed=False)
    class Before:
        @T.prim_func
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32 + 1 - 2,), "float32")
            A[T.min(T.int64(1), n)] = T.float32(0)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.prim_func
        def main(a: T.handle):
            n = T.int64()
            A = T.match_buffer(a, (n * 32 + 1 - 2,), "float32")
            A[T.int64(1)] = T.float32(0)

    after = tvm.tir.transform.Simplify()(Before)
    tvm.ir.assert_structural_equal(after["main"], Expected["main"])


def test_nested_if_elimination():
    @T.prim_func(private=True)
    def before(a: T.Buffer((2, 8), "int32"), b: T.Buffer((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(
                i0 == 1 and 6 <= j0, 0, T.max(0, T.if_then_else(i0 == 1 and 6 <= j0, 0, a[i0, j0]))
            )

    @T.prim_func(private=True)
    def expected(a: T.Buffer((2, 8), "int32"), b: T.Buffer((2, 8), "int32")):
        for i0, j0 in T.grid(2, 8):
            b[i0, j0] = T.if_then_else(i0 == 1 and 6 <= j0, 0, T.max(0, a[i0, j0]))

    after = _apply_simplify(before)
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
