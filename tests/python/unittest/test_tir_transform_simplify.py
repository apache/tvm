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

    def transform(self):
        def inner(mod):
            config = {
                "tir.Simplify": {
                    "transitively_prove_inequalities": self.transitively_prove_inequalities,
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


if __name__ == "__main__":
    tvm.testing.main()
