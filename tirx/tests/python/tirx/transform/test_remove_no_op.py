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


@pytest.mark.parametrize("case", ["empty_nest", "side_effect", "zero_extent"])
def test_remove_no_op(case):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "int64"), m: T.int32, n: T.int32, i: T.int32):
            if T.constexpr(case == "empty_nest"):
                for ii in T.serial(4):
                    for j in T.serial(n):
                        for k in T.serial(m):
                            if ii * m + j + k < n:
                                T.evaluate(m)
                            else:
                                T.evaluate(n)
            elif T.constexpr(case == "side_effect"):
                T.evaluate(1)
                A[i + 1] = A[i] + T.int64(1)
                T.evaluate(1)
            else:
                for j in T.serial(0):
                    A[j + 1] = A[j] + T.int64(1)

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16,), "int64"), m: T.int32, n: T.int32, i: T.int32):
            if T.constexpr(case == "side_effect"):
                A[i + 1] = A[i] + T.int64(1)
            else:
                T.evaluate(0)

    After = tvm_tirx.tirx.transform.RemoveNoOp()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_remove_no_op_with_invalid_extent():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
            for i in T.serial(16):
                for j in T.serial(i - 20):
                    B[i] = A[i] + j

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
            T.evaluate(0)

    After = tvm_tirx.tirx.transform.RemoveNoOp()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def _apply_remove_no_op(func, max_simplification_steps=0):
    """Helper function to apply RemoveNoOp transform with config."""

    @I.ir_module
    class Before:
        main = func

    config = {
        "tirx.RemoveNoOp": {
            "max_simplification_steps": max_simplification_steps,
        }
    }
    with tvm_tirx.ir.transform.PassContext(config=config):
        mod = tvm_tirx.tirx.transform.RemoveNoOp()(Before)
    return mod["main"]


def test_remove_empty_for_loop():
    """A for-loop whose body is a no-op is itself a no-op."""

    @T.function(private=True)
    def before():
        for i in T.serial(16):
            T.evaluate(0)

    @T.function(private=True)
    def expected():
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_zero_extent_loop():
    """A for-loop with no extent is a no-op."""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(0):
            A[i] = 42

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_unused_let():
    """With flat Bind, unused bindings are preserved by remove_no_op.

    Unused Bind elimination requires a separate two-pass approach
    and is not handled by the current remove_no_op pass.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        x = 5
        for i in T.serial(16):
            A[i] = 0

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        x = 5
        for i in T.serial(16):
            A[i] = 0

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_let_used_only_in_no_op():
    """With flat Bind, unused bindings are preserved by remove_no_op.

    The zero-extent for loop is removed, but the Bind for x remains
    since unused Bind elimination is not handled by remove_no_op.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        x = 5
        for i in T.serial(0):
            A[i] = x

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        x = 5
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_keep_side_effects_of_let():
    """Side-effect Bind is preserved as-is by remove_no_op."""

    @T.function(private=True)
    def before():
        x = T.call_extern("extern_func", ty="int32")
        T.evaluate(0)

    @T.function(private=True)
    def expected():
        x = T.call_extern("extern_func", ty="int32")
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_empty_then_case():
    """A no-op then_case can be removed."""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                T.evaluate(0)
            else:
                A[i] = 42

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if not (i < 8):
                A[i] = 42

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_empty_else_case():
    """A no-op else_case can be removed."""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42
            else:
                T.evaluate(0)

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if i < 8:
                A[i] = 42

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_suppress_removal_of_unused_write():
    """Sequential writes to the same location are not removed.

    Dataflow analysis is no longer supported.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            A[i] = 100
            A[i] = 42

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_keep_first_write_when_used():
    """For two sequential writes, keep the first if it is used"""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            A[i] = 100
            A[i] = A[i] + 1

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_keep_partially_overwritten_loop():
    """Keep partially overwritten regions

    If the second loop doesn't entirely overwrite the first, the first
    may not be removed be kept.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            A[i] = 100

        for i in T.serial(16):
            if i < 12:
                A[i] = 42

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_remove_read_write():
    """Writing a value to the same location as was just read is a no-op."""

    @T.function(private=True)
    def before(A: T.Tensor(1, "int32")):
        A[0] = A[0]

    @T.function(private=True)
    def expected(A: T.Tensor(1, "int32")):
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_keep_read_write_to_different_indices():
    """Writing a value to a different index should not be removed"""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(15):
            A[i] = A[i + 1]

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_remove_read_write_same_index_different_expression():
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    store can be removed. With flat Bind, the Bind for i and the
    enclosing loops remain since unused Bind elimination is not
    handled by remove_no_op.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for io, ii in T.grid(4, 4):
            i: T.let[T.int32] = 4 * io + ii
            A[4 * io + ii] = A[i]

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        for io in range(4):
            for ii in range(4):
                i: T.let[T.int32] = 4 * io + ii

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_remove_read_write_same_index_using_constraint():
    """Writing a value to the same location as the read is a no-op.

    If the value of the index can be proven to be the same, then the
    no-op can be removed.  This may require using the a constraint
    that is known from a conditional containing the read/write.
    """

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]
            else:
                A[i] = A[0]

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            if i != 0:
                A[i] = A[i - 1]

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.xfail(reason="Dead alloc removal not yet implemented for flat AllocTensor")
def test_remove_empty_temporary():
    """An allocation with a no-op body is a no-op."""

    @T.function(private=True)
    def before():
        A = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="local")])
        T.evaluate(0)

    @T.function(private=True)
    def expected():
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.xfail(reason="Dead alloc removal not yet implemented for flat AllocTensor")
def test_remove_empty_temporary_with_decl_tensor():
    """Remove DeclTensor alongside Allocate

    If an unused allocation is removed, any DeclTensor instances that
    refer to it should also be removed.
    """

    @T.function(private=True)
    def before():
        A = T.alloc_tensor(ty_args=[T.Tensor([4, 4], "int32", scope="local")])
        A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(16, "int32", scope="local")])
        T.evaluate(0)

    @T.function(private=True)
    def expected():
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_unused_temporary():
    """An unused allocation is a no-op."""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32")):
        B = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="local")])
        for i in T.serial(16):
            A[i] = 1

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32")):
        for i in T.serial(16):
            A[i] = 1

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_unused_write_into_temporary():
    """A write that only impacts a temporary allocation is a no-op."""

    @T.function(private=True)
    def before():
        A = T.alloc_tensor(ty_args=[T.Tensor([16], "int32", scope="local")])
        for i in T.serial(16):
            A[i] = 0

    @T.function(private=True)
    def expected():
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_keep_used_write_into_temporary():
    """A write into a temporary that is used later must be kept."""

    @T.function(private=True)
    def before(B: T.Tensor(16, "int32")):
        A = T.alloc_tensor(ty_args=[T.Tensor([16], "int32", scope="local")])
        for i in T.serial(16):
            A[i] = 0

        for i in T.serial(16):
            B[i] = A[i]

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


@pytest.mark.xfail(reason="Not implemented yet")
def test_remove_write_into_temporary():
    """A write that only impacts a temporary allocation is a no-op."""

    @T.function(private=True)
    def before(A: T.Tensor(16, "int32"), C: T.Tensor(1, "int32")):
        B = T.alloc_tensor(ty_args=[T.Tensor([16], "int32", scope="local")])
        for i in T.serial(16):
            B[i] = A[i]

        C[0] = 0
        for i in T.serial(16):
            C[0] = C[0] + B[i]

        for i in T.serial(16):
            B[i] = 0

    @T.function(private=True)
    def expected(A: T.Tensor(16, "int32"), C: T.Tensor(1, "int32")):
        B = T.alloc_tensor(ty_args=[T.Tensor([16], "int32", scope="local")])
        for i in T.serial(16):
            B[i] = A[i]

        C[0] = 0
        for i in T.serial(16):
            C[0] = C[0] + B[i]

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_certain_condition():
    """The conditon of the If-Else node is certain.
    This would cause `Segmentation fault` error before."""

    @T.function(private=True)
    def before():
        if True:
            T.evaluate(0)
        else:
            T.evaluate(0)

    @T.function(private=True)
    def expected():
        T.evaluate(0)

    after = _apply_remove_no_op(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
