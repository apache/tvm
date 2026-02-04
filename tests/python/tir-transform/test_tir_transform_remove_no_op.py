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
from tvm.script import tir as T
import tvm.testing

import pytest


def nop():
    return tvm.tir.Evaluate(1)


def test_remove_no_op():
    i = te.var("i")
    j = te.var("j")
    k = te.var("k")
    m = te.var("m")
    n = te.var("n")
    dtype = "int64"
    Ab = tvm.tir.decl_buffer((n,), dtype)
    stmt = tvm.tir.For(
        i,
        0,
        4,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.For(
            j,
            0,
            n,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.For(
                k,
                0,
                m,
                tvm.tir.ForKind.SERIAL,
                tvm.tir.IfThenElse((i * m + j + k < n), tvm.tir.Evaluate(m), tvm.tir.Evaluate(n)),
            ),
        ),
    )

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body

    assert isinstance(ret, tvm.tir.Evaluate)
    store = tvm.tir.BufferStore(Ab, tvm.tir.BufferLoad(Ab, [i]) + 1, [i + 1])
    stmt2 = tvm.tir.SeqStmt([nop(), tvm.tir.SeqStmt([store, nop()])])

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt2))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert ret == store

    # remove zero extent loop
    stmt3 = tvm.tir.For(i, 0, 0, tvm.tir.ForKind.SERIAL, store)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab], stmt3))
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


def test_remove_no_op_with_invalid_extent():
    @T.prim_func
    def main(A: T.Buffer((16), "int32"), B: T.Buffer((16), "int32")) -> None:
        for i in T.serial(16):
            for j in T.serial(i - 20):
                B[i] = A[i] + j

    mod = tvm.ir.module.IRModule.from_expr(main)
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    assert isinstance(ret, tvm.tir.Evaluate)


def _apply_remove_no_op(mod, use_dataflow_analysis=False, max_simplification_steps=0):
    """Helper function to apply RemoveNoOp transform with config."""
    config = {
        "tir.RemoveNoOp": {
            "use_dataflow_analysis": use_dataflow_analysis,
            "max_simplification_steps": max_simplification_steps,
        }
    }
    with tvm.transform.PassContext(config=config):
        mod = tvm.tir.transform.RemoveNoOp()(mod)
    return mod


def test_remove_empty_for_loop():
    """A for-loop whose body is a no-op is itself a no-op."""

    @T.prim_func(private=True)
    def before():
        for i in T.serial(16):
            T.evaluate(0)

    @T.prim_func(private=True)
    def expected():
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_zero_extent_loop():
    """A for-loop with no extent is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(0):
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_unused_let():
    """A let statement that is never used is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        x = 5
        for i in T.serial(16):
            A[i] = 0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 0

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_let_used_only_in_no_op():
    """A let statement that is never used is a no-op.

    Similar to test_remove_unused_let, but the usage of the let binding
    may have been removed by an earlier removal of another no-op.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        x = 5
        for i in T.serial(0):
            A[i] = x

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_side_effects_of_let():
    """The side effects of a no-op let must be kept."""

    @T.prim_func(private=True)
    def before():
        x = T.call_extern("extern_func", dtype="int32")
        T.evaluate(0)

    @T.prim_func(private=True)
    def expected():
        T.evaluate(T.call_extern("extern_func", dtype="int32"))

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_empty_then_case():
    """A no-op then_case can be removed."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                T.evaluate(0)
            else:
                A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if not (i < 8):
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_empty_else_case():
    """A no-op else_case can be removed."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42
            else:
                T.evaluate(0)

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_unused_write():
    """For two sequential writes, the first is a no-op"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_suppress_removal_of_unused_write():
    """Dataflow analysis requires the config to opt-in

    Like test_remove_unused_write, but dataflow analysis isn't enabled.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=False)
    tvm.ir.assert_structural_equal(mod["main"], before)


def test_keep_side_effects_of_unused_write():
    """For two sequential writes, the first value may have side effects"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = T.call_extern("extern_func", dtype="int32")
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            T.evaluate(T.call_extern("extern_func", dtype="int32"))
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_first_write_when_used():
    """For two sequential writes, keep the first if it is used"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100
            A[i] = A[i] + 1

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], before)


def test_remove_overwritten_loop():
    """Remove repeated writes to the same region

    If two loops write to the same region, the first is a no-op.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_overwritten_subloop():
    """Remove repeated writes to the same region

    If the first loop writes to a subset of the region, the first loop
    is a no-op.  Similar to test_remove_overwritten_loop, but the first
    loop's extents are a subset of the second loop.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(4, 12):
            A[i] = 100

        for i in T.serial(16):
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_partially_overwritten_loop():
    """Keep partially overwritten regions

    If the second loop doesn't entirely overwrite the first, the first
    may not be removed be kept.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], before)


def test_remove_overwritten_predicated_loop_with_identical_condition():
    """Remove repeated writes to the same predicated region.

    Similar to test_keep_partially_overwritten_loop, except the first loop
    has the same predicate as the second, and can therefore be
    removed.

    In the past, this test has had performance regressions in which
    the runtime increased from a few seconds to nearly ten minutes.
    The "max_simplification_steps" parameter is set at twice the
    current number of steps required, in order to prevent similar
    performance regression.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 100

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True, max_simplification_steps=200000)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_overwritten_predicated_loop_with_provable_condition():
    """Remove repeated writes to the same predicated region.

    Similar to
    test_remove_overwritten_predicated_loop_with_identical_condition, except
    the first loop's predicate is not a precise match for the second
    loop's predicate.  So long as the regions written in the first
    loop are a subset of those written in the second loop, they can be
    removed.

    In the past, this test has had performance regressions in which
    the runtime increased from a few seconds to nearly ten minutes.
    The "max_simplification_steps" parameter is set at twice the
    current number of steps required, in order to prevent similar
    performance regression.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 10:
                A[i] = 100

        for i in T.serial(16):
            if i // 4 < 3:
                A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i // 4 < 3:
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True, max_simplification_steps=200000)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_separated_overwrites():
    """Remove repeated writes to the same predicated region.

    Similar to test_remove_overwritten_loop, but with an
    independent loop between the first and second write of the buffer.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32"), B: T.Buffer(16, "int32")):
        for i in T.serial(16):
            B[i] = 0

        for i in T.serial(16):
            A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_separated_overwrite_of_predicated_loop():
    """Remove repeated writes to the same predicated region.

    Similar to test_remove_separated_overwrites, but the independent loop
    between the first and second writes to a different subset
    of the same buffer.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 100

        for i in T.serial(16):
            if i > 12:
                A[i] = 15

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i > 12:
                A[i] = 15

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_read_write():
    """Writing a value to the same location as was just read is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(1, "int32")):
        A[0] = A[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer(1, "int32")):
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_read_write_to_different_indices():
    """Writing a value to a different index should not be removed"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(15):
            A[i] = A[i + 1]

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], before)


def test_remove_read_write_same_index_different_expression():
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    no-op can be removed, even if they have different forms of the
    expression.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for io, ii in T.grid(4, 4):
            i = 4 * io + ii
            A[4 * io + ii] = A[i]

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_read_write_same_index_using_constraint():
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    no-op can be removed.  This may require using the a constraint
    that is known from a conditional containing the read/write.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]
            else:
                A[i] = A[0]

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_writing_of_known_value():
    """Writing a value that already exists at that index is a no-op"""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

        A[4] = 4

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_one_of_duplicate_loops():
    """Must not reason based on a touch point after removing it.

    If the first loop is removed because it is overwritten by the
    second loop, and the second loop is removed because it writes the
    same value as the first loop, the overall transformation is no
    longer valid.  In this case, only one of the two should be
    removed.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

        for i in T.serial(16):
            A[i] = i

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = i

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod, use_dataflow_analysis=True)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_empty_temporary():
    """An allocation with a no-op body is a no-op."""

    @T.prim_func(private=True)
    def before():
        A = T.allocate([16], "int32", "local")
        T.evaluate(0)

    @T.prim_func(private=True)
    def expected():
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_remove_empty_temporary_with_decl_buffer():
    """Remove DeclBuffer alongside Allocate

    If an unused allocation is removed, any DeclBuffer instances that
    refer to it should also be removed.
    """

    @T.prim_func(private=True)
    def before():
        A = T.decl_buffer([4, 4], "int32", scope="local")
        A_flat = T.decl_buffer(16, "int32", scope="local", data=A.data)
        T.evaluate(0)

    @T.prim_func(private=True)
    def expected():
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_unused_temporary():
    """An unused allocation is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32")):
        B = T.allocate([16], "int32", "local")
        for i in T.serial(16):
            A[i] = 1

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 1

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_unused_write_into_temporary():
    """A write that only impacts a temporary allocation is a no-op."""

    @T.prim_func(private=True)
    def before():
        A = T.decl_buffer([16], "int32", scope="local")
        for i in T.serial(16):
            A[i] = 0

    @T.prim_func(private=True)
    def expected():
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_keep_used_write_into_temporary():
    """A write into a temporary that is used later must be kept."""

    @T.prim_func(private=True)
    def before(B: T.Buffer(16, "int32")):
        A = T.decl_buffer([16], "int32", scope="local")
        for i in T.serial(16):
            A[i] = 0

        for i in T.serial(16):
            B[i] = A[i]

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], before)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_write_into_temporary():
    """A write that only impacts a temporary allocation is a no-op."""

    @T.prim_func(private=True)
    def before(A: T.Buffer(16, "int32"), C: T.Buffer(1, "int32")):
        B = T.decl_buffer([16], "int32", scope="local")
        for i in T.serial(16):
            B[i] = A[i]

        C[0] = 0
        for i in T.serial(16):
            C[0] = C[0] + B[i]

        for i in T.serial(16):
            B[i] = 0

    @T.prim_func(private=True)
    def expected(A: T.Buffer(16, "int32"), C: T.Buffer(1, "int32")):
        B = T.decl_buffer([16], "int32", scope="local")
        for i in T.serial(16):
            B[i] = A[i]

        C[0] = 0
        for i in T.serial(16):
            C[0] = C[0] + B[i]

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_certain_condition():
    """The conditon of the If-Else node is certain.
    This would cause `Segmentation fault` error before."""

    @T.prim_func(private=True)
    def before():
        if True:
            T.evaluate(0)
        else:
            T.evaluate(0)

    @T.prim_func(private=True)
    def expected():
        T.evaluate(0)

    mod = tvm.IRModule.from_expr(before)
    mod = _apply_remove_no_op(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
