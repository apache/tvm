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
from tvm.script import tir as T


def test_stmt_simplify():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = te.size_var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.if_scope(i < 12):
            A[i] = C[i]

    body = tvm.tir.LetStmt(n, 10, ib.get())
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, C, n], body))
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body, tvm.tir.BufferStore)


def test_thread_extent_simplify():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = te.size_var("n")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    ib.scope_attr(tx, "thread_extent", n)
    ib.scope_attr(tx, "thread_extent", n)
    ib.scope_attr(ty, "thread_extent", 1)
    with ib.if_scope(tx + ty < 12):
        A[tx] = C[tx + ty]
    body = tvm.tir.LetStmt(n, 10, ib.get())
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, C, n], body))
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body.body.body, tvm.tir.BufferStore)


def test_if_likely():
    ib = tvm.tir.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    n = te.size_var("n")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    ib.scope_attr(tx, "thread_extent", 32)
    ib.scope_attr(ty, "thread_extent", 32)
    with ib.if_scope(ib.likely(tx * 32 + ty < n)):
        with ib.if_scope(ib.likely(tx * 32 + ty < n)):
            A[tx] = C[tx * 32 + ty]
    body = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([A, C, n], body))
    body = tvm.tir.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body.body, tvm.tir.IfThenElse)
    assert not isinstance(body.body.body.then_case, tvm.tir.IfThenElse)


def test_basic_likely_elimination():
    n = te.size_var("n")
    X = te.placeholder(shape=(n,), name="x")
    W = te.placeholder(shape=(n + 1,), dtype="int32", name="w")

    def f(i):
        start = W[i]
        extent = W[i + 1] - W[i]
        rv = te.reduce_axis((0, extent))
        return te.sum(X[rv + start], axis=rv)

    Y = te.compute(X.shape, f, name="y")
    s = te.create_schedule([Y.op])
    stmt = tvm.lower(s, [X, W, Y], simple_mode=True)
    assert "if" not in str(stmt)


def test_complex_likely_elimination():
    def cumsum(X):
        """
        Y[i] = sum(X[:i])
        """
        (m,) = X.shape
        s_state = te.placeholder((m + 1,), dtype="int32", name="state")
        s_init = te.compute((1,), lambda _: tvm.tir.const(0, "int32"))
        s_update = te.compute((m + 1,), lambda l: s_state[l - 1] + X[l - 1])
        return tvm.te.scan(s_init, s_update, s_state, inputs=[X], name="cumsum")

    def sparse_lengths_sum(data, indices, lengths):
        oshape = list(data.shape)
        oshape[0] = lengths.shape[0]
        length_offsets = cumsum(lengths)

        def sls(n, d):
            gg = te.reduce_axis((0, lengths[n]))
            indices_idx = length_offsets[n] + gg
            data_idx = indices[indices_idx]
            data_val = data[data_idx, d]
            return te.sum(data_val, axis=gg)

        return te.compute(oshape, sls)

    m, n, d, i, l = (
        te.size_var("m"),
        te.size_var("n"),
        te.size_var("d"),
        te.size_var("i"),
        te.size_var("l"),
    )
    data_ph = te.placeholder((m, d * 32), name="data")
    indices_ph = te.placeholder((i,), name="indices", dtype="int32")
    lengths_ph = te.placeholder((n,), name="lengths", dtype="int32")
    Y = sparse_lengths_sum(data_ph, indices_ph, lengths_ph)
    s = te.create_schedule([Y.op])
    (n, d) = s[Y].op.axis
    (do, di) = s[Y].split(d, factor=32)
    (gg,) = s[Y].op.reduce_axis
    s[Y].reorder(n, do, gg, di)
    s[Y].vectorize(di)
    stmt = tvm.lower(s, [data_ph, indices_ph, lengths_ph, Y], simple_mode=True)
    assert "if" not in str(stmt)


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    transitively_prove_inequalities = False
    convert_boolean_to_and_of_ors = False
    apply_constraints_to_boolean_branches = False
    propagate_knowns_to_prove_conditional = False
    propagate_knowns_to_simplify_expressions = False

    def transform(self):
        def inner(mod):
            config = {
                "tir.Simplify": {
                    "transitively_prove_inequalities": self.transitively_prove_inequalities,
                    "convert_boolean_to_and_of_ors": self.convert_boolean_to_and_of_ors,
                    "apply_constraints_to_boolean_branches": self.apply_constraints_to_boolean_branches,
                    "propagate_knowns_to_prove_conditional": self.propagate_knowns_to_prove_conditional,
                    "propagate_knowns_to_simplify_expressions": self.propagate_knowns_to_simplify_expressions,
                }
            }
            with tvm.transform.PassContext(config=config):
                mod = tvm.tir.transform.Simplify()(mod)
            return mod

        return inner


class TestLoadStoreNoop(BaseBeforeAfter):
    """Store of a value that was just read from the same location is a no-op."""

    def before(A: T.Buffer[(1,), "float32"]):
        A[0] = A[0]

    def expected(A: T.Buffer[(1,), "float32"]):
        T.evaluate(0)


class TestLoadStoreNoopAfterSimplify(BaseBeforeAfter):
    """As test_load_store_noop, but requiring simplification to identify.

    Previously, a bug caused the self-assignment of a buffer to
    checked based on the pre-simplification assignment, not the
    post-simplification.  This test is to identify any similar
    regression.
    """

    def before(A: T.Buffer[(1,), "float32"]):
        A[0] = A[0] + (5.0 - 5.0)

    def expected(A: T.Buffer[(1,), "float32"]):
        T.evaluate(0)


class TestNestedCondition(BaseBeforeAfter):
    """Nested IfThenElse with the same condition can be simplified.

    Requires const_int_bound to narrow scope of i within the
    conditional, or for rewrite_simplify to recognize the literal
    constraint.
    """

    def before(A: T.Buffer[(16,), "float32"]):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    def expected(A: T.Buffer[(16,), "float32"]):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0


class TestNestedProvableCondition(BaseBeforeAfter):
    """Simplify inner conditional using constraint from outer.

    Requires const_int_bound to narrow scope of i within the
    conditional.
    """

    def before(A: T.Buffer[(16,), "float32"]):
        for i in T.serial(16):
            if i == 5:
                if i < 7:
                    A[i] = 0.0

    def expected(A: T.Buffer[(16,), "float32"]):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0


class TestNestedVarCondition(BaseBeforeAfter):
    """Simplify inner conditional using constraint from outer.

    Requires for rewrite_simplify to recognize the repeated
    constraint.
    """

    def before(A: T.Buffer[(16,), "float32"], n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i == n:
                    A[i] = 0.0

    def expected(A: T.Buffer[(16,), "float32"], n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 0.0


class TestAlteredBufferContents(BaseBeforeAfter):
    """No simplification of data-dependent conditionals.

    A literal constraint must not be propagated if the values
    referenced may change.  TIR requires single assignment of
    variables, so Var objects may be assumed constant, but BufferLoad
    may not.
    """

    def before(A: T.Buffer[(1,), "int32"], n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0

    expected = before


class TestNegationOfCondition(BaseBeforeAfter):
    """Use negation of outer condition to simplify innner.

    Within the body of an if statement, the negation of the
    condition is known to be false.
    """

    def before(A: T.Buffer[(16,), "int32"]):
        for i in T.serial(16):
            if i == 5:
                if i != 5:
                    A[i] = 0
                else:
                    A[i] = 1

    def expected(A: T.Buffer[(16,), "int32"]):
        for i in T.serial(16):
            if i == 5:
                A[i] = 1


class TestNegationOfNotEqual(BaseBeforeAfter):
    """As TestNegationOfVarCondition, but with a != outer condition.

    Because ConstIntBoundAnalyzer only tracks the min and max allowed
    values, the outer i!=5 condition does provide a constraint on the
    bounds.  This test relies on RewriteSimplifier to recognize
    ``i==5`` as the negation of a literal constraint.
    """

    def before(A: T.Buffer[(16,), "int32"]):
        for i in T.serial(16):
            if i != 5:
                if i == 5:
                    A[i] = 0
                else:
                    A[i] = 1

    def expected(A: T.Buffer[(16,), "int32"]):
        for i in T.serial(16):
            if i != 5:
                A[i] = 1


class TestNegationOfVarCondition(BaseBeforeAfter):
    """As TestNegationOfVarCondition, but with a dynamic condition.

    This simplification cannot be done with ConstIntBoundAnalyzer, and
    must rely on RewriteSimplifier recognizing the repeated literal.
    """

    def before(A: T.Buffer[(16,), "int32"], n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i != n:
                    A[i] = 0
                else:
                    A[i] = 1

    def expected(A: T.Buffer[(16,), "int32"], n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 1


class TestLiteralConstraintSplitBooleanAnd(BaseBeforeAfter):
    """Split a boolean AND into independent constraints

    A single if condition may impose multiple literal constraints.
    Each constraint that is ANDed together to form the condition
    should be treated as an independent constraint.  The use of n in
    the condition is to ensure we exercise RewriteSimplifier.
    """

    def before(A: T.Buffer[(16, 16), "int32"], n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                if i == n:
                    A[i, j] = 0

    def expected(A: T.Buffer[(16, 16), "int32"], n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                A[i, j] = 0


class TestLiteralConstraintSplitBooleanOr(BaseBeforeAfter):
    """Split a boolean OR into independent constraints

    Similar to TestLiteralConstraintSplitBooleanAnd, but splitting a
    boolean OR into independent conditions.  This uses the
    simplification that ``!(x || y) == !x && !y``.

    The use of ``n`` in the condition is to ensure we exercise
    RewriteSimplifier.
    """

    def before(A: T.Buffer[(16, 16), "int32"], n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2

    def expected(A: T.Buffer[(16, 16), "int32"], n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                A[i, j] = 2


class TestProveConditionUsingLet(BaseBeforeAfter):
    """Simplify conditions using non-inlined let bindings

    Not all let bindings are inlined when they occur in later
    expressions.  However, even if they are not inlined, they may be
    used to prove the value of a condition.
    """

    @T.prim_func
    def before(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            if condition or i >= 3:
                A[i] = condition

    @T.prim_func
    def expected(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            A[i] = condition


class TestProveLetCondition(BaseBeforeAfter):
    """Simplify conditions using non-inlined let bindings

    Not all let bindings are inlined when they occur in later
    expressions.  However, even if they are not inlined, they may be
    used to prove the value of a condition.
    """

    @T.prim_func
    def before(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            if i < 3:
                if condition:
                    A[i] = condition

    @T.prim_func
    def expected(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            if i < 3:
                A[i] = condition


class TestProveRepeatedLetCondition(BaseBeforeAfter):
    """Simplify conditions using non-inlined let bindings

    A variable may be used as a literal constraint, and be recognized
    as being True within the context of the constraint.
    """

    @T.prim_func
    def before(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                if condition:
                    A[i] = condition

    @T.prim_func
    def expected(A: T.Buffer[4, "bool"]):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                A[i] = True


class TestIfThenElseExpr(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0, dtype="float32")

    @T.prim_func
    def expected(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0


class TestCeilLog2Int(BaseBeforeAfter):
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.prim_func
    def before(A: T.Buffer[1, "int32"]):
        A[0] = T.cast(
            T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32"
        )

    @T.prim_func
    def expected(A: T.Buffer[1, "int32"]):
        A[0] = 4


class TestLeftCeilLog2LowerBound(BaseBeforeAfter):
    """Integer bounds are propagated through topi.math.ceil_log2"""

    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            x = T.cast(
                T.ceil(T.log2(T.cast(i + 1024 + 1, "float64"), dtype="float64"), dtype="float64"),
                dtype="int32",
            )
            if x == 11:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            A[i] = 0.0


class TestLeftShiftLowerBound(BaseBeforeAfter):
    """Integer bounds are propagated through left shift

    min(1 << i) = 1 << min(i)
                = 1 << 0
                = 1
    """

    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if T.shift_left(1, i, dtype="int32") >= 1:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            A[i] = 0.0


class TestLeftShiftUpperBound(BaseBeforeAfter):
    """Integer bounds are propagated through left shift

    max(31 << i) = 31 << max(i)
                 = 31 << 15
                 = 1015808
    """

    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if T.shift_left(31, i, dtype="int32") <= 1015808:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            A[i] = 0.0


class TestLeftShiftOfNegativeValue(BaseBeforeAfter):
    """No const int bounds of left shift of negative value.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if -64 <= T.shift_left(-i, 4, dtype="int32"):
                A[i] = 0.0

    expected = before


class TestLeftShiftByNegativeValue(BaseBeforeAfter):
    """No const int bounds of left shift by negative bit count.

    This is target dependent, and does not currently have a specified
    behavior in TIR.  For example, in CodeGenC, this generates C code
    with undefined behavior.
    """

    @T.prim_func
    def before(A: T.Buffer[16, "float32"]):
        for i in T.serial(16):
            if T.shift_left(16, -i, dtype="int32") <= 16:
                A[i] = 0.0

    expected = before


class TestRemoveTransitivelyProvableCondition(BaseBeforeAfter):
    """Remove comparisons that may be proven using multiple others

    For example, the `0 < i` and `i <= j` conditions can be used to prove
    that `0 < j`.
    """

    transitively_prove_inequalities = True

    i, j, k = [tvm.tir.Var(name, "int32") for name in "ijk"]
    zero = tvm.tir.IntImm("int32", 0)

    test_case = tvm.testing.parameter(
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
    )

    @tvm.testing.fixture
    def before(self, test_case):
        priors, postulate, _ = test_case

        @T.prim_func
        def func(A: T.Buffer[1, "bool"]):
            if priors:
                A[0] = postulate

        return func

    @tvm.testing.fixture
    def expected(self, test_case):
        priors, postulate, provable = test_case

        analyzer = tvm.arith.Analyzer()
        priors = analyzer.canonical_simplify(priors)

        if provable:

            @T.prim_func
            def func(A: T.Buffer[1, "bool"]):
                if priors:
                    A[0] = True

            return func

        else:
            postulate = analyzer.canonical_simplify(postulate)

            @T.prim_func
            def func(A: T.Buffer[1, "bool"]):
                if priors:
                    A[0] = postulate

            return func


class TestSuppressTransitivelyProvableCondition(BaseBeforeAfter):
    transitively_prove_inequalities = False

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        if i < j and j < k:
            A[0] = i < k

    expected = before


class TestRewriteAsAndOfOrs(BaseBeforeAfter):
    """If enabled, rewrite boolean expressions into AND of OR"""

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[3, "bool"]):
        T.evaluate(A[0] or (A[1] and A[2]))

    def expected(A: T.Buffer[3, "bool"]):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))


class TestSuppressRewriteAsAndOfOrs(BaseBeforeAfter):
    """Only rewrite into AND of OR when allowed"""

    convert_boolean_to_and_of_ors = False

    def before(A: T.Buffer[3, "bool"]):
        T.evaluate(A[0] or (A[1] and A[2]))

    expected = before


class TestRewriteAsAndOfOrsWithTopLevelAnd(BaseBeforeAfter):
    """The expression being rewritten may start with an AND

    Like TestRewriteAsAndOfOrs, but with an AndNode as the outermost
    booelan operator.  Even though it is primarily OR nodes that are
    being rewritten, the call to SimplifyAsAndOfOrs should apply to
    the outermost AndNode or OrNode in order to enable better
    simplification.
    """

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[4, "bool"]):
        T.evaluate((A[0] or A[1]) and (A[1] or (A[0] and A[2] and A[3])))

    def expected(A: T.Buffer[4, "bool"]):
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


class TestRewriteAsAndOfOrsWithSimplificationBetweenGroups(BaseBeforeAfter):
    """Apply rewrite rules between OR groups that differ by a single element

    The expression `(k==20 and k!=30)` could be rewritten into `(k==20)`.
    However, by default these two terms must appear as part of an explict part
    of the simplified expression.  The AndOfOr simplification checks for
    rewrite patterns of the form `(A or B) and (A or C)`, where `(B and C)` can
    simplify to a single expression `D`.  These can be rewritten to `(A or D)`.
    """

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (i == 0 or j == 10 or k != 30)

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = i == 0 or j == 10 or k == 20


class TestRewriteAsAndOfOrsWithSimplificationBetweenReorderedGroups(BaseBeforeAfter):
    """Rewrite rules between OR groups do not depend on order

    Like TestRewriteAsAndOfOrsWithSimplificationBetweenGroups, but the groups
    are ordered differently.  If this removes a group entirely, the result is
    ordered according to the first group in the expression.
    """

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10 or k == 20) and (j == 10 or k != 30 or i == 0)

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = j == 10 or k == 20 or i == 0


class TestRewriteAsAndOfOrUsingSimplificationAcrossAnd(BaseBeforeAfter):
    """Apply AndNode rewrites to non-adjacent expressions

    The RewriteSimplifier rules only check for simplifications between
    left/right branches of an And/Or node.  Simplifications that would require
    rearranging components in a chain of And/Or nodes are not performed.
    """

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (k == 20) and ((i == 0 or j == 10) and (k != 30))

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 0 or j == 10) and (k == 20)


class TestRewriteAsAndOfOrUsingSimplificationWithinOr(BaseBeforeAfter):
    """Rewrite rules between OR groups do not depend on order

    The RewriteSimplifier rules only check for simplifications between
    left/right branches of an And/Or node.  Simplifications that would require
    rearranging components in a chain of And/Or nodes are not performed.

    This test validates that `(i == 20) or (i != 30)` can be rewritten to
    `(i != 30)`, even when there's an intervening clause between the
    clauses being simplified.
    """

    convert_boolean_to_and_of_ors = True

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (i == 20) or (j == 0) or (i != 30)

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32, k: T.int32):
        A[0] = (j == 0) or (i != 30)


class TestConditionalFloorMod(BaseBeforeAfter):
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

    def before(A: T.Buffer[1, "bool"], i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0

    def expected(A: T.Buffer[1, "bool"], i: T.int32):
        if T.floormod(i, -2) == 0:
            A[0] = True


class TestSimplifyRHSOfBooleanAndUsingLHS(BaseBeforeAfter):
    """Boolean expressions can introduce contexts.

    In `A and B`, the result of `B` only matters when `A` is
    true, and can be simplified under that context.  This test
    simplifies `n < 10` under the assumption that `n < 5`.
    """

    apply_constraints_to_boolean_branches = True

    def before(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 5 and n < 10

    def expected(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 5


class TestSimplifyLHSOfBooleanAndUsingRHS(BaseBeforeAfter):
    """Boolean expressions can introduce contexts for their arguments.

    Like TestSimplifyRHSOfBooleanAndUsingLHS, but using the RHS to
    simplify the LHS.
    """

    apply_constraints_to_boolean_branches = True

    def before(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 10 and n < 5

    def expected(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 5


class TestSimplifyRHSOfBooleanOrUsingLHS(BaseBeforeAfter):
    """Boolean expressions can introduce contexts.

    In `A or B`, the result of `B` only matters when `A` is false, so
    `B` can be simplified under the assumption that `A` is false.
    This test simplifies `n < 5` under the assumption that `!(n < 10)`
    """

    apply_constraints_to_boolean_branches = True

    def before(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 10 or n < 5

    def expected(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 10


class TestSimplifyLHSOfBooleanOrUsingRHS(BaseBeforeAfter):
    """Boolean expressions can introduce contexts for their arguments.

    Like TestSimplifyRHSOfBooleanOrUsingLHS, but using the RHS to
    simplify the LHS.
    """

    apply_constraints_to_boolean_branches = True

    def before(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 5 or n < 10

    def expected(A: T.Buffer[1, "bool"], n: T.int32):
        A[0] = n < 10


class TestSimplifyRHSOfBooleanAndUsingLHSWithoutConst(BaseBeforeAfter):
    """Boolean expressions can introduce contexts.

    Like TestSimplifyRHSOfBooleanAndUsingLHS, but with variables in
    the conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    apply_constraints_to_boolean_branches = True
    transitively_prove_inequalities = True

    def before(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 5 and n < m + 10

    def expected(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 5


class TestSimplifyLHSOfBooleanAndUsingRHSWithoutConst(BaseBeforeAfter):
    """Boolean expressions can introduce contexts for their arguments.

    Like TestSimplifyLHSOfBooleanAndUsingRHS, but with variables in
    the conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    apply_constraints_to_boolean_branches = True
    transitively_prove_inequalities = True

    def before(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 10 and n < m + 5

    def expected(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 5


class TestSimplifyRHSOfBooleanOrUsingLHSWithoutConst(BaseBeforeAfter):
    """Boolean expressions can introduce contexts.

    Like TestSimplifyRHSOfBooleanOrUsingLHS, but with variables in the
    conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    apply_constraints_to_boolean_branches = True
    transitively_prove_inequalities = True

    def before(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 10 or n < m + 5

    def expected(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 10


class TestSimplifyLHSOfBooleanOrUsingRHSWithoutConst(BaseBeforeAfter):
    """Boolean expressions can introduce contexts for their arguments.

    Like TestSimplifyLHSOfBooleanOrUsingRHS, but with variables in the
    conditions, preventing ConstIntBoundAnalyzer from handling it.
    This proof requires the extension to transitively prove
    inequalities.
    """

    apply_constraints_to_boolean_branches = True
    transitively_prove_inequalities = True

    def before(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 5 or n < m + 10

    def expected(A: T.Buffer[1, "bool"], n: T.int32, m: T.int32):
        A[0] = n < m + 10


class TestProvableConditionWithOffset(BaseBeforeAfter):
    """Use scoped-constraint to prove inequalities"""

    transitively_prove_inequalities = False

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32):
        if i < j:
            A[0] = i < j + 1

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32):
        if i < j:
            A[0] = True


class TestMostRestrictiveConditional(BaseBeforeAfter):
    """Preferentially prove part of a compound conditional.

    Even if we cannot prove a conditional as true or false on its own,
    proving that a conditional must satisfy a stronger condition may
    allow for later rewrites.  For example, if it is known that `a <= b`,
    then `a >= b` cannot be proven, but can be reduced to `a == b`.
    """

    i, j, k = [tvm.tir.Var(name, "int32") for name in "ijk"]
    tir_int = tvm.tir.IntImm("int32", 0)

    test_case = tvm.testing.parameter(
        (i <= tir_int, tir_int <= i, i == tir_int),
        (i <= tir_int, i != tir_int, i < tir_int),
        (i != tir_int, i <= tir_int, i < tir_int),
        (i != tir_int, tir_int <= i, tir_int < i),
        (i <= j, j <= i, j == i),
        (i <= j, i != j, i < j),
        (i != j, i <= j, i < j),
        (i != j, j <= i, j < i),
    )

    @tvm.testing.fixture
    def before(self, test_case):
        priors, expr_before, _ = test_case

        @T.prim_func
        def func(A: T.Buffer[1, "bool"]):
            if priors:
                A[0] = expr_before

        return func

    @tvm.testing.fixture
    def expected(self, test_case):
        priors, _, expr_after = test_case

        @T.prim_func
        def func(A: T.Buffer[1, "bool"]):
            if priors:
                A[0] = expr_after

        return func


class TestProvableConditionWithOffset(BaseBeforeAfter):
    """Use scoped-constraint to prove inequalities"""

    transitively_prove_inequalities = False

    def before(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32):
        if i < j:
            A[0] = i < j + 1

    def expected(A: T.Buffer[1, "bool"], i: T.int32, j: T.int32):
        if i < j:
            A[0] = True


class TestAlteredBufferContents(BaseBeforeAfter):
    """Propagation of data-dependent conditionals.

    A literal constraint must not be propagated if the values
    referenced may change.  TIR requires single assignment of
    variables, so Var objects may be assumed constant, but BufferLoad
    may not.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[(1,), "int32"], n: T.int32):
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

    def expected(A: T.Buffer[(1,), "int32"], n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            A[0] = 10


class TestPossiblyAlteredBufferContents(BaseBeforeAfter):
    """No simplification of data-dependent conditionals.

    Like TestAlteredBufferContents, but the `m==0` conditional
    prevents the value of `A[0]` from being known at the point of the
    inner conditional, either as `A[0] == n` from the outer
    conditional or as `A[0] == n+1` from the write statement.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[(1,), "int32"], n: T.int32, m: T.int32):
        if A[0] == n:
            if m == 0:
                A[0] = A[0] + 1

            if A[0] == n:
                A[0] = 5
            else:
                A[0] = 10

    expected = before


class TestSimplifyInputAssumption(BaseBeforeAfter):
    """A T.assume annotation may be used to simplify"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"], n: T.int32):
        T.evaluate(T.assume(n == 0))
        if n == 0:
            A[0] = 42

    def expected(A: T.Buffer[1, "int32"], n: T.int32):
        T.evaluate(T.assume(n == 0))
        A[0] = 42


class TestSimplifyInputAssumption(BaseBeforeAfter):
    """A T.assume annotation may be used to simplify"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"], n: T.int32):
        T.evaluate(T.assume(n == 0))
        if n == 0:
            A[0] = 42

    def expected(A: T.Buffer[1, "int32"], n: T.int32):
        T.evaluate(T.assume(n == 0))
        A[0] = 42


class TestNoSimplifyFromScopedInputAssumption(BaseBeforeAfter):
    """A T.assume inside a scope may not apply outside that scope"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"], n: T.int32, m: T.int32):
        if m == 0:
            T.evaluate(T.assume(n == 0))

        if n == 0:
            A[0] = 42

    expected = before


class TestSimplifyConditionalUsingBufferValue(BaseBeforeAfter):
    """Simplify a conditional using the known value in the buffer"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"]):
        A[0] = 0

        if A[0] == 0:
            A[0] = 42

    def expected(A: T.Buffer[1, "int32"]):
        A[0] = 0
        A[0] = 42


class TestKeepExpressionSimplifyUsingBufferValue(BaseBeforeAfter):
    """Do not simplify expressions in general using known values in the buffer

    For now, because this is equivalent to inlining, preventing this
    usage from occurring.  Known buffer values may be used to prove
    conditionals, but should not be used for other simplifications.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"], B: T.Buffer[1, "int32"]):
        A[0] = 0
        B[0] = A[0]

    expected = before


class TestSimplifyConditionalInLoopUsingBufferValue(BaseBeforeAfter):
    """Simplify a conditional using the known value in the buffer

    Like TestSimplifyConditionalUsingBufferValue, but the value used
    to simplify is set in a previous loop.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i

        for j in T.serial(16):
            if A[j] == j:
                B[j] = 42
            else:
                B[j] = 100

    def expected(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            A[i] = i

        for j in T.serial(16):
            B[j] = 42


class TestSimplifyUsingBufferAssumption(BaseBeforeAfter):
    """A T.assume may apply to a buffer's contents"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"]):
        T.evaluate(T.assume(A[0] == 0))

        if A[0] == 0:
            A[0] = 42

    def expected(A: T.Buffer[1, "int32"]):
        T.evaluate(T.assume(A[0] == 0))
        A[0] = 42


class TestSimplifyUsingBufferAssumptionInLoop(BaseBeforeAfter):
    """An assumption about buffer contents may apply to a range"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == i))

        for i in T.serial(16):
            if A[i] < 100:
                A[i] = 0

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == i))

        for i in T.serial(16):
            A[i] = 0


class TestSimplifyUsingPartiallyKnownBufferConditional(BaseBeforeAfter):
    """An assumption about buffer contents may apply to only part of a buffer"""

    propagate_knowns_to_prove_conditional = True
    apply_constraints_to_boolean_branches = True

    def before(A: T.Buffer[16, "int32"]):
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

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if 14 <= i:
                T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                A[i] = 42

            else:
                if A[i] == 0:
                    A[i] = 100


class TestSimplifyUsingPartiallyKnownBufferExpression(BaseBeforeAfter):
    """An assumption about buffer contents may apply to only part of a buffer

    Like TestSimplifyUsingPartiallyKnownBufferConditional, but the
    conditional is expressed as part of T.assume, instead of in the
    control flow.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                if A[i] == 0:
                    A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        for i in T.serial(16):
            if 14 <= i:
                A[i] = 42


class TestNoSimplificationIfPredicateNotMet(BaseBeforeAfter):
    """Assumptions about buffer contents must apply to all cases to be used

    Like TestSimplifyUsingPartialBufferAssumptionInLoop, but the
    predicate in the second loop does not match the predicate in the
    first loop.  Therefore, the `T.assume` refers to a different set
    of indices.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if 14 <= i:
                T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if i < 14:
                if A[i] == 0:
                    A[i] = 42

    expected = before


class TestNoSimplifyUsingInvalidatedScopedConstraint(BaseBeforeAfter):
    """A write may not be used for proofs outside its conditional"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            if i == 0:
                A[i] = 0

            if A[i] == 0:
                A[i] = 42

    expected = before


class TestNoSimplifyUsingOverwrittenValue(BaseBeforeAfter):
    """A write that may have been overwritten may not be treated as known

    The appearance of "A[i] = 5" must prevent the earlier constraint
    from being used for simplification.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42

    expected = before


class TestNoSimplifyUsingLoopDependentBufferValue(BaseBeforeAfter):
    """Do not simplify assuming reads are invariant

    If a buffer's value changes across loop iterations, the buffer's
    value before the loop should not be used to simplify conditionals
    within the loop.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"], B: T.Buffer[1, "int32"]):
        B[0] = 0
        for i in T.serial(16):
            if B[0] < 10:
                B[0] = A[i] * 2 + B[0]
            else:
                B[0] = A[i] + B[0]

    expected = before


class TestSimplifyPriorToOverwrittenValue(BaseBeforeAfter):
    """A known value may be used until it is overwritten

    Like TestNoSimplifyUsingOverwrittenValue, but the use of the
    known `A[i]` value occurs before it is overwritten.

    Like TestNoSimplifyUsingLoopDependentBufferValue, but the loop
    iterations are all independent.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            if A[i] == 0:
                A[i] = 17

            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42

    def expected(A: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            T.evaluate(T.assume(A[i] == 0))

        for i in T.serial(16):
            A[i] = 17

            if i == 0:
                A[i] = 5

            if A[i] == 0:
                A[i] = 42


class TestSimplifyElementWiseUsingPreLoopBufferValue(BaseBeforeAfter):
    """Allow data-Do not simplify assuming reads are invariant

    If an element-wise loop reads and overwrites a buffer value, the
    pre-loop buffer value may be used to simplify conditions that
    occur prior to the write.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            if B[i] < 10:
                B[i] = A[i] * 2 + B[i]
            else:
                B[i] = A[i] + B[i]

    def expected(A: T.Buffer[16, "int32"], B: T.Buffer[16, "int32"]):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            B[i] = A[i] * 2 + B[i]


class TestSimplifyNonConditional(BaseBeforeAfter):
    """Propagate a known value to later expressions."""

    propagate_knowns_to_simplify_expressions = True

    def before(A: T.Buffer[1, "int32"]):
        A[0] = 0
        A[0] = A[0] + 1

    def expected(A: T.Buffer[1, "int32"]):
        A[0] = 0
        A[0] = 1


class TestSuppressSimplifyNonConditional(BaseBeforeAfter):
    """Propagate a known value to later expressions.

    Like TestSimplifyNonConditional, but with data-propagation turned off.
    """

    propagate_knowns_to_simplify_expressions = False

    def before(A: T.Buffer[1, "int32"]):
        A[0] = 0
        A[0] = A[0] + 1

    expected = before


class TestSimplifyUsingTransitiveKnownBufferValue(BaseBeforeAfter):
    """Propagate known buffer values

    If a known value of a buffer depends on another known value, it
    can be tracked backwards through both.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[1, "int32"]):
        T.evaluate(T.assume(A[0] == 0))

        A[0] = A[0] + 1
        A[0] = A[0] + 1
        A[0] = A[0] + 1

        if A[0] == 3:
            A[0] = 42

    def expected(A: T.Buffer[1, "int32"]):
        T.evaluate(T.assume(A[0] == 0))

        A[0] = A[0] + 1
        A[0] = A[0] + 1
        A[0] = A[0] + 1

        A[0] = 42


class TestSimplifyRampIndexBroadcastValue(BaseBeforeAfter):
    """Simplifications involving buffer loads with ramp indices"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[4, "int32"]):
        A[T.ramp(0, 1, 4)] = T.broadcast(0, 4)

        if A[0] == 0:
            A[0] = 42

        if A[1] == 0:
            A[1] = 60

    def expected(A: T.Buffer[4, "int32"]):
        A[T.ramp(0, 1, 4)] = T.broadcast(0, 4)

        A[0] = 42
        A[1] = 60


class TestSimplifyRampIndexRampValue(BaseBeforeAfter):
    """Simplifications involving buffer loads with ramp indices"""

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[4, "int32"]):
        A[T.ramp(0, 1, 4)] = T.ramp(11, 1, 4)

        if A[0] == 11:
            A[0] = 42

        if A[1] == 12:
            A[1] = 60

    def expected(A: T.Buffer[4, "int32"]):
        A[T.ramp(0, 1, 4)] = T.ramp(11, 1, 4)

        A[0] = 42
        A[1] = 60


class TestSimplifyUsingPartiallyProvenBufferValueGather(BaseBeforeAfter):
    """Propagate known buffer values in part of buffer.

    Even if a constraint can't be solved for all values in an
    assignment, it may be provable in part of a buffer.  Here, the
    known 0 values in the padding of A produces known 0 values in the
    padding of B.
    """

    transitively_prove_inequalities = True
    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[24, "int32"], B: T.Buffer[24, "int32"], F: T.Buffer[3, "int32"]):
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

    def expected(A: T.Buffer[24, "int32"], B: T.Buffer[24, "int32"], F: T.Buffer[3, "int32"]):
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


class TestSimplifyUsingPartiallyProvenBufferValueScatter(BaseBeforeAfter):
    """Propagate known buffer values in part of buffer.

    Like TestSimplifyUsingPartiallyProvenBufferValueGather, but the
    compute loop is over the input buffer A, rather than the output
    buffer B.
    """

    propagate_knowns_to_prove_conditional = True

    def before(A: T.Buffer[24, "int32"], B: T.Buffer[24, "int32"], F: T.Buffer[3, "int32"]):
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

    def expected(A: T.Buffer[24, "int32"], B: T.Buffer[24, "int32"], F: T.Buffer[3, "int32"]):
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


class TestSimplifyBufferStore(BaseBeforeAfter):
    """Simplification using prior known"""

    propagate_knowns_to_simplify_expressions = True

    def before(A: T.Buffer[1, "int32"]):
        A[0] = 5
        A[0] = A[0] + 7

    def expected(A: T.Buffer[1, "int32"]):
        A[0] = 5
        A[0] = 12


if __name__ == "__main__":
    tvm.testing.main()
