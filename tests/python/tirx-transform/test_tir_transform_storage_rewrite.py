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
# ruff: noqa: F401, F841
import sys

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def test_alloc_seq():
    scope_tb = "local.L0A"

    @T.prim_func
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_buffer((200,), scope=scope_tb)
                A[j] = T.float32(1.2)
            for j in range(10):
                B = T.alloc_buffer((200,), scope=scope_tb)
                B[j] = T.float32(1.3)

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            num_alloc[0] += 1
            assert n.buffer.shape[0].value == 200

    tvm.tirx.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_alloc_different_dtypes():
    # Test cross-loop buffer access with buffers allocated in parent scope
    def make_mod(dtype_list, length):
        assert len(dtype_list) == 4

        @T.prim_func
        def func():
            # Allocate all buffers in parent scope (before any loops)
            A = T.alloc_buffer((length,), dtype_list[0], scope="local.L0A")
            B = T.alloc_buffer((length,), dtype_list[1], scope="local.L0A")
            C = T.alloc_buffer((length,), dtype_list[2], scope="local.L0A")
            D = T.alloc_buffer((length,), dtype_list[3], scope="local.L0A")
            E = T.alloc_buffer((length,), "int8", scope="local.L0A")

            for j in range(length):
                A[j] = T.Cast(dtype_list[0], 1)
            for j in range(length):
                B[j] = T.Cast(dtype_list[1], 1)
            for j in range(length):
                C[j] = T.Cast(dtype_list[2], 1)
            for j in range(length):
                D[j] = T.Cast(dtype_list[3], 1)
            for j in range(length):
                E[j] = (
                    T.Cast("int8", A[j])
                    + T.Cast("int8", B[j])
                    + T.Cast("int8", C[j])
                    + T.Cast("int8", D[j])
                )

        return tvm.IRModule.from_expr(func)

    def dtype_bit_len(dtype):
        index = 0
        for i in dtype:
            if i.isdigit():
                break
            index += 1
        return int(dtype[index:])

    def offset_generater(dtype_list, length):
        dtype_len_list = [dtype_bit_len(i) for i in dtype_list]
        base_len = dtype_len_list[0]
        return sum([i * length / base_len for i in dtype_len_list])

    def dtype_test(dtype_list, length):
        def verify(n):
            if isinstance(n, tvm.tirx.AllocBuffer):
                assert n.buffer.shape[0].value == offset

        mod = make_mod(dtype_list, length)
        offset = offset_generater(dtype_list, length)

        body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body
        tvm.tirx.stmt_functor.post_order_visit(body, verify)

    length = 1024
    dtype_list = ["float16", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float32", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["float64", "int32", "uint16", "int8"]
    dtype_test(dtype_list, length)

    dtype_list = ["int8", "int32", "uint16", "uint8"]
    dtype_test(dtype_list, length)


def test_address_of():
    # In this test, the storage rewrite pass is allowed to
    # combine buffers B and D, but not C
    @T.prim_func
    def before(A: T.Buffer(8, "float32"), E: T.Buffer(8, "float32")):
        B = T.alloc_buffer((8,))
        for i in range(8):
            B[i] = (
                T.call_extern("deref", T.address_of(A[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(A[0]), dtype="float32")
                + T.float32(1)
            )
        C = T.alloc_buffer((8,))
        for i in range(8):
            C[i] = (
                T.call_extern("deref", T.address_of(B[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(B[0]), dtype="float32")
                + T.float32(2)
            )
        D = T.alloc_buffer((8,))
        for i in range(8):
            D[i] = (
                T.call_extern("deref", T.address_of(C[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(C[0]), dtype="float32")
                + T.float32(2)
            )
        for i in range(8):
            E[i] = (
                T.call_extern("deref", T.address_of(D[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(D[0]), dtype="float32")
                + T.float32(3)
            )

    def verify(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            total_alloc[0] += n.buffer.shape[0].value

    total_alloc = [0]
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, verify)
    assert total_alloc[0] == 24

    total_alloc[0] = 0
    mod = tvm.tirx.transform.StorageRewrite()(mod)
    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, verify)
    assert total_alloc[0] == 16


def test_parallel_alloc():
    @T.prim_func
    def func1(n: T.int32):
        for i in T.parallel(n):
            for j in range(10):
                A = T.alloc_buffer((n,))
                A[j] = A[j] + T.float32(2)

    mod = tvm.IRModule.from_expr(func1)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func1"]

    # With flat AllocBuffer, the for body is a SeqStmt; first element is AllocBuffer
    assert isinstance(body.body.body[0], tvm.tirx.AllocBuffer)

    @T.prim_func
    def func2(n: T.int32):
        for t in T.serial(n):
            with T.attr(T.int32(1), "pragma_scope", "parallel_launch_point"):
                for i in T.parallel(n):
                    for j in range(10):
                        A = T.alloc_buffer((n,))
                        A[j] = A[j] + T.float32(2)

    mod = tvm.IRModule.from_expr(func2)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func2"]

    assert isinstance(body.body.body.body.body[0], tvm.tirx.AllocBuffer)


def test_while_alloc():
    @T.prim_func
    def func_parallel(n: T.int32):
        for i in T.parallel(n):
            j = T.alloc_buffer((1,), "int32")
            j[0] = 0
            while j[0] < 10:
                A = T.alloc_buffer((n,))
                A[j[0]] = A[j[0]] + T.float32(2)
                j[0] = j[0] + j[0] + 1

    @T.prim_func
    def func_serial(n: T.int32):
        for i in T.serial(n):
            j = T.alloc_buffer((1,), "int32")
            j[0] = 0
            while j[0] < 10:
                A = T.alloc_buffer((n,))
                A[j[0]] = A[j[0]] + T.float32(2)
                j[0] = j[0] + j[0] + 1

    mod = tvm.IRModule.from_expr(func_parallel)
    # parallel (i, 0, n) {
    #   allocate j[int32 * 1]
    #   j[0] = 0
    #   while((j[0] < 10)){
    #     // attr [A] storage_scope = "global"
    #     allocate A[float32 * n]
    #     A[j[0]] = (A[j[0]] + 2f)
    #     j[0] = (j[0] + (j[0] + 1))
    #   }
    # }
    body = tvm.tirx.transform.StorageRewrite()(mod)["func_parallel"]
    # Navigate to inside the for loop, then check that allocations exist
    # The structure with DeclBuffer is:
    #   parallel (i, 0, n) { DeclBuffer(j, DeclBuffer(A, ...)) }
    # or with Allocate+DeclBuffer pairs
    inner = body.body.body  # inside For
    # Skip DeclBuffer nodes to find Allocate
    num_alloc = [0]

    def count_alloc(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            num_alloc[0] += 1

    tvm.tirx.stmt_functor.post_order_visit(inner, count_alloc)
    assert num_alloc[0] == 2  # j and A allocations

    mod = tvm.IRModule.from_expr(func_serial)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func_serial"]
    num_alloc[0] = 0
    tvm.tirx.stmt_functor.post_order_visit(body.body, count_alloc)
    assert num_alloc[0] == 2  # j and A allocations


def test_alloc_seq_type():
    @T.prim_func
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_buffer((200,), scope="local.L0A")
                A1 = T.alloc_buffer((200,), scope="local.L0A")
                A[j] = T.float32(1.2)
                A1[j] = T.float32(1.3)
                B = T.alloc_buffer((200,), "int16", scope="local.L0A")
                B[j] = T.int16(1)
                C = T.alloc_buffer((200,), "int16", scope="local.L0A")
                C[j] = T.int16(1)
                D = T.alloc_buffer((200,), "int16", scope="local.L0A")
                D[j] = B[j] + C[j]
                A2 = T.alloc_buffer((200,), scope="local.L0A")
                A2[j] = A[j]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            num_alloc[0] += 1
            assert n.buffer.shape[0].value == 500

    tvm.tirx.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_alloc_seq_type2():
    scope_tb = "local.L0A2"

    @T.prim_func
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_buffer((200,), scope=scope_tb)
                A[j] = T.float32(1.2)
            for j in range(20):
                B = T.alloc_buffer((400,), "int16", scope=scope_tb)
                B[j] = T.int16(1)
            for j in range(10):
                C = T.alloc_buffer((200,), scope=scope_tb)
                C[j] = T.float32(1.2)

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            num_alloc[0] += 1
            assert n.buffer.shape[0].value == 200

    tvm.tirx.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_reuse_small_buffer():
    @T.prim_func
    def func(n: T.int32):
        for i in T.serial(n):
            for j in range(10):
                A = T.alloc_buffer((200,), "int16", scope="local.L0A")
                A[j] = T.int16(1)
                B = T.alloc_buffer((200,), "int16", scope="local.L0A")
                B[j] = T.int16(1)
                B1 = T.alloc_buffer((200,), "int16", scope="local.L0A")
                B1[j] = A[j] + B[j]
                C = T.alloc_buffer((400,), "int16", scope="local.L0A")
                C[j] = T.int16(1)
                D = T.alloc_buffer((400,), "int16", scope="local.L0A")
                D[j] = T.int16(1)
                E = T.alloc_buffer((400,), "int16", scope="local.L0A")
                E[j] = C[j]

    mod = tvm.IRModule.from_expr(func)
    body = tvm.tirx.transform.StorageRewrite()(mod)["func"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tirx.AllocBuffer):
            num_alloc[0] += 1
            assert n.buffer.shape[0].value == 800

    tvm.tirx.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_access_in_let_value():
    @T.prim_func
    def func(A: T.Buffer((8,), "float32")):
        for i in range(8):
            B = T.alloc_buffer((1,))
            B[0] = 3.14
            x: T.float32 = T.exp(B[0], dtype="float32")
            A[i] = (x + 1.0) / (x - 1.0)

    @T.prim_func
    def func_rewritten(A: T.Buffer((8,), "float32")) -> None:
        B = T.alloc_buffer((1,))
        for i in range(8):
            B[0] = 3.14
            x: T.float32 = T.exp(B[0], dtype="float32")
            A[i] = (x + 1.0) / (x - 1.0)

    mod = tvm.tirx.transform.StorageRewrite()(
        tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    )
    tvm.ir.assert_structural_equal(mod["main"], func_rewritten.with_attr("global_symbol", "main"))


def test_let_buffer_rewrite():
    """StorageRewrite replaces the bound var of backing allocations

    If StorageRewrite replaces the backing variable of an array, such
    as when vectorizing the storage type, the variable must be
    replaced in the Bind that defines it.  Currently, StmtMutator
    only visits usage of variables, and does not visit definitions of
    variables, so the definition in a Bind must be explicitly
    handled.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main() -> None:
            A_data: T.handle("int32") = T.call_extern("dummy_func", dtype="handle")
            A = T.decl_buffer([8], "int32", data=A_data)
            A[0:8] = T.broadcast(42, 8)

    @I.ir_module(check_well_formed=False)
    class Expected:
        @T.prim_func
        def main() -> None:
            A_data: T.handle("int32x8") = T.call_extern("dummy_func", dtype="handle")
            A = T.decl_buffer([8], "int32", data=A_data)
            A_1 = T.Buffer([1], "int32x8", data=A_data)
            A_1[0] = T.broadcast(42, 8)

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_rewrite_in_place_use_of_non_flat_buffer():
    """A non-flat buffer may be re-used for in-place operations"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
            B = T.decl_buffer(
                [16, 16],
                dtype="float32",
                axis_separators=[1],
            )
            C = T.decl_buffer(
                [16, 16],
                dtype="float32",
                axis_separators=[1],
            )

            for i, j in T.grid(16, 16):
                B[i, j] = A[i, j]

            for i, j in T.grid(16, 16):
                C[i, j] = 2.0 * B[i, j]

            for i, j in T.grid(16, 16):
                D[i, j] = C[i, j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
            B = T.decl_buffer([16, 16], dtype="float32", axis_separators=[1])
            C = T.decl_buffer(
                [16, 16],
                dtype="float32",
                axis_separators=[1],
                data=B.data,
            )

            for i, j in T.grid(16, 16):
                B[i, j] = A[i, j]

            for i, j in T.grid(16, 16):
                C[i, j] = 2.0 * B[i, j]

            for i, j in T.grid(16, 16):
                D[i, j] = C[i, j]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_no_rewrite_of_shared_non_flat_buffer():
    """In general, sharing of non-flat buffer isn't supported

    The current packing algorithms in StorageRewrite assume a flat
    memory space, and do not support packing of N-d buffers.  For
    buffers with axis separators, normal buffer sharing should be
    disabled.

    Like test_rewrite_in_place_use_of_non_flat_buffer, except that B and C do
    not have matching shapes.
    """

    @T.prim_func
    def Before(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
        B = T.decl_buffer(
            [16, 16],
            dtype="float32",
            axis_separators=[1],
        )
        C = T.decl_buffer(
            [20, 20],
            dtype="float32",
            axis_separators=[1],
        )

        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

        for i, j in T.grid(16, 16):
            C[i, j] = 2.0 * B[i, j]

        for i, j in T.grid(16, 16):
            D[i, j] = C[i, j]

    Expected = Before

    After = tvm.tirx.transform.StorageRewrite()(tvm.IRModule.from_expr(Before))
    tvm.ir.assert_structural_equal(After["Before"], Expected)


def test_rewrite_decl_buffer():
    """A DeclBuffer node may appear in StorageRewrite's input"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
            B = T.decl_buffer(16, dtype="float32")
            C = T.decl_buffer(16, dtype="float32")

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
            B = T.decl_buffer(16, dtype="float32")
            C = T.decl_buffer(16, dtype="float32", data=B.data)

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_no_orphaned_decl_buffer():
    """A DeclBuffer of an unused Allocate should be removed

    StorageRewrite removes any allocations that are unused.  When it
    does so, any DeclBuffer that refers to that allocation should also
    be removed.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
            B = T.decl_buffer(16, dtype="float32")
            C = T.decl_buffer(16, dtype="float32")
            Unused = T.decl_buffer(16, dtype="float32")

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
            B = T.decl_buffer(16, dtype="float32")
            C = T.decl_buffer(16, dtype="float32", data=B.data)

            for i in range(16):
                B[i] = A[i]

            for i in range(16):
                C[i] = 2.0 * B[i]

            for i in range(16):
                D[i] = C[i]

    After = tvm.tirx.transform.StorageRewrite()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
