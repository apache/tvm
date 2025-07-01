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
import sys

import pytest

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T


def register_mem(scope_tb, max_bits):
    # Register mem
    @tvm.register_func("tvm.info.mem.%s" % scope_tb)
    def mem_info_inp_buffer():
        return tvm.ir.make_node(
            "target.MemoryInfo",
            unit_bits=16,
            max_simd_bits=32,
            max_num_bits=max_bits,
            head_address=None,
        )


def test_alloc_seq():
    scope_tb = "local.L0A"
    max_bits = 1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)

    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope=scope_tb)
            A[j] = 1.2
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="B", scope=scope_tb)
            A[j] = 1.3

    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 200

    tvm.tir.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_alloc_different_dtypes():
    def stmt_generater(dtype_list, length):
        ib = tvm.tir.ir_builder.create()
        base_dtype = dtype_list[0]
        global_a = te.placeholder((length,), name="global_a", dtype=base_dtype)
        assert len(dtype_list) == 4
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[0]
            A = ib.allocate(dtype, length, name="A", scope="local.L0A")
            A[j] = tvm.tir.const(1, dtype=dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[1]
            B = ib.allocate(dtype, length, name="B", scope="local.L0A")
            B[j] = tvm.tir.const(1, dtype=dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[2]
            C = ib.allocate(dtype, length, name="C", scope="local.L0A")
            C[j] = tvm.tir.const(1, dtype=dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = dtype_list[3]
            D = ib.allocate(dtype, length, name="D", scope="local.L0A")
            D[j] = tvm.tir.const(1, dtype=dtype)
        with ib.for_range(0, length, name="j") as j:
            dtype = "int8"
            E = ib.allocate(dtype, length, name="E", scope="local.L0A")
            E[j] = A[j].astype(dtype) + B[j].astype(dtype) + C[j].astype(dtype) + D[j].astype(dtype)
        return ib.get()

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
            if isinstance(n, tvm.tir.Allocate):
                assert n.extents[0].value == offset

        body = stmt_generater(dtype_list, length)
        offset = offset_generater(dtype_list, length)

        mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], body))
        body = tvm.tir.transform.StorageRewrite()(mod)["main"].body

        tvm.tir.stmt_functor.post_order_visit(body, verify)

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
        B_data = T.allocate([8], "float32")
        B = T.Buffer(8, data=B_data, align=32)
        for i in range(8):
            B[i] = (
                T.call_extern("deref", T.address_of(A[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(A[0]), dtype="float32")
                + T.float32(1)
            )
        C_data = T.allocate([8], "float32")
        C = T.Buffer(8, data=C_data, align=32)
        for i in range(8):
            C[i] = (
                T.call_extern("deref", T.address_of(B[i]), dtype="float32")
                + T.call_extern("deref", T.address_of(B[0]), dtype="float32")
                + T.float32(2)
            )
        D_data = T.allocate([8], "float32")
        D = T.Buffer(8, data=D_data, align=32)
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
        if isinstance(n, tvm.tir.Allocate):
            total_alloc[0] += n.extents[0].value

    total_alloc = [0]
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod.show()
    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, verify)
    assert total_alloc[0] == 24

    total_alloc[0] = 0
    mod = tvm.tir.transform.StorageRewrite()(mod)
    mod.show()
    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, verify)
    assert total_alloc[0] == 16


def test_parallel_alloc():
    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i", kind="parallel") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", n, name="A", scope="global")
            A[j] = A[j] + 2

    body = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"]

    assert isinstance(body.body.body, tvm.tir.Allocate)

    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="t") as i:
        ib.scope_attr(
            tvm.tir.const(1, "int32"), "pragma_scope", tvm.tir.StringImm("parallel_launch_point")
        )
        with ib.for_range(0, n, name="i", kind="parallel") as i:
            with ib.for_range(0, 10, name="j") as j:
                A = ib.allocate("float32", n, name="A", scope="global")
                A[j] = A[j] + 2
    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"]

    assert isinstance(body.body.body.body.body, tvm.tir.Allocate)


def test_while_alloc():
    def get_mod(kind="serial"):
        ib = tvm.tir.ir_builder.create()
        n = te.var("n")
        with ib.for_range(0, n, name="i", kind=kind) as i:
            j = ib.allocate("int32", 1, name="j", scope="global")
            j[0] = 0
            with ib.while_loop(j[0] < 10):
                A = ib.allocate("float32", n, name="A", scope="global")
                A[j[0]] = A[j[0]] + 2
                j[0] += j[0] + 1

        body = ib.get()
        return tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))

    mod = get_mod(kind="parallel")
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
    body = tvm.tir.transform.StorageRewrite()(mod)["main"]
    # parallel (i, 0, n) {
    #   allocate j[int32 * 1]
    #   allocate A[float32 * n]
    #   j[0] = 0
    #   while((j[0] < 10)){
    #     A[j[0]] = (A[j[0]] + 2f)
    #     j[0] = (j[0] + (j[0] + 1))
    #   }
    # }
    assert isinstance(body.body.body, tvm.tir.Allocate)  # j
    assert isinstance(body.body.body.body, tvm.tir.Allocate)  # A

    mod = get_mod(kind="serial")
    # for (i, 0, n) {
    #   allocate j[int32 * 1]
    #   j[0] = 0
    #   while((j[0] < 10)){
    #     // attr [A] storage_scope = "global"
    #     allocate A[float32 * n]
    #     A[j[0]] = (A[j[0]] + 2f)
    #     j[0] = (j[0] + (j[0] + 1))
    #   }
    # }
    body = tvm.tir.transform.StorageRewrite()(mod)["main"]
    # allocate j[int32 * 1]
    # allocate A[float32 * n]
    # for (i, 0, n) {
    #   j[0] = 0
    #   while((j[0] < 10)){
    #     A[j[0]] = (A[j[0]] + 2f)
    #     j[0] = (j[0] + (j[0] + 1))
    #   }
    # }
    assert isinstance(body.body, tvm.tir.Allocate)  # j
    assert isinstance(body.body.body, tvm.tir.Allocate)  # A


def test_alloc_seq_type():
    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope="local.L0A")
            A1 = ib.allocate("float32", 200, name="A1", scope="local.L0A")
            A[j] = 1.2
            A1[j] = 1.3
            B = ib.allocate("int16", 200, name="B", scope="local.L0A")
            B[j] = tvm.tir.const(1, "int16")
            C = ib.allocate("int16", 200, name="C", scope="local.L0A")
            C[j] = tvm.tir.const(1, "int16")
            D = ib.allocate("int16", 200, name="D", scope="local.L0A")
            D[j] = B[j] + C[j]
            A2 = ib.allocate("float32", 200, name="A2", scope="local.L0A")
            A2[j] = A[j]

    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 500

    tvm.tir.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_alloc_seq_type2():
    scope_tb = "local.L0A2"
    max_bits = 1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)

    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("float32", 200, name="A", scope=scope_tb)
            A[j] = 1.2
        with ib.for_range(0, 20, name="j") as j:
            B = ib.allocate("int16", 400, name="B", scope=scope_tb)
            B[j] = tvm.tir.const(1, "int16")
        with ib.for_range(0, 10, name="j") as j:
            C = ib.allocate("float32", 200, name="C", scope=scope_tb)
            C[j] = 1.2

    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 200

    tvm.tir.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_reuse_small_buffer():
    ib = tvm.tir.ir_builder.create()
    n = te.var("n")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            A = ib.allocate("int16", 200, name="A", scope="local.L0A")
            A[j] = tvm.tir.const(1, "int16")
            B = ib.allocate("int16", 200, name="B", scope="local.L0A")
            B[j] = tvm.tir.const(1, "int16")
            B1 = ib.allocate("int16", 200, name="B1", scope="local.L0A")
            B1[j] = A[j] + B[j]
            C = ib.allocate("int16", 400, name="C", scope="local.L0A")
            C[j] = tvm.tir.const(1, "int16")
            D = ib.allocate("int16", 400, name="D", scope="local.L0A")
            D[j] = tvm.tir.const(1, "int16")
            E = ib.allocate("int16", 400, name="E", scope="local.L0A")
            E[j] = C[j]

    body = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], body))
    body = tvm.tir.transform.StorageRewrite()(mod)["main"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 800

    tvm.tir.stmt_functor.post_order_visit(body, verify)
    assert num_alloc[0] == 1


def test_access_in_let_value():
    @T.prim_func
    def func(A: T.Buffer((8,), "float32")):
        for i in range(8):
            B_data = T.allocate((1,), "float32", "global")
            B = T.Buffer(shape=[1], dtype="float32", data=B_data)
            B[0] = 3.14
            x: T.float32 = T.exp(B[0], dtype="float32")
            A[i] = (x + 1.0) / (x - 1.0)

    @T.prim_func
    def func_rewritten(A: T.Buffer((8,), "float32")) -> None:
        B_data = T.allocate((1,), "float32", "global")
        B = T.Buffer(shape=[1], dtype="float32", data=B_data)
        for i in range(8):
            B[0] = 3.14
            x: T.float32 = T.exp(B[0], dtype="float32")
            A[i] = (x + 1.0) / (x - 1.0)

    mod = tvm.tir.transform.StorageRewrite()(
        tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    )
    tvm.ir.assert_structural_equal(mod["main"], func_rewritten.with_attr("global_symbol", "main"))


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.StorageRewrite()


class TestLetBufferRewrite(BaseCompare):
    """StorageRewrite replaces the bound var of backing allocations

    If StorageRewrite replaces the backing variable of an array, such
    as when vectorizing the storage type, the variable must be
    replaced in the LetStmt that defines it.  Currently, StmtMutator
    only visits usage of variables, and does not visit definitions of
    variables, so the definition in a LetStmt must be explicitly
    handled.
    """

    def before() -> None:
        A_data: T.handle("int32") = T.call_extern("dummy_func", dtype="handle")
        A = T.Buffer([8], "int32", data=A_data)
        A[0:8] = T.broadcast(42, 8)

    def expected() -> None:
        A_data: T.handle("int32x8") = T.call_extern("dummy_func", dtype="handle")
        A = T.Buffer([1], "int32x8", data=A_data)
        A[0] = T.broadcast(42, 8)


class TestRewriteInPlaceUseOfNonFlatBuffer(BaseCompare):
    """A non-flat buffer may be re-used for in-place operations"""

    def before(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
        B_data = T.allocate(
            [16, 16],
            dtype="float32",
            scope="global",
        )
        B = T.Buffer(
            [16, 16],
            dtype="float32",
            axis_separators=[1],
            data=B_data,
        )
        C_data = T.allocate(
            [16, 16],
            dtype="float32",
            scope="global",
        )
        C = T.Buffer(
            [16, 16],
            dtype="float32",
            axis_separators=[1],
            data=C_data,
        )

        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

        for i, j in T.grid(16, 16):
            C[i, j] = 2.0 * B[i, j]

        for i, j in T.grid(16, 16):
            D[i, j] = C[i, j]

    def expected(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
        B_data = T.allocate(
            [16, 16],
            dtype="float32",
            scope="global",
        )
        B = T.Buffer([16, 16], dtype="float32", axis_separators=[1], data=B_data)
        C = T.Buffer(
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


class TestNoRewriteOfSharedNonFlatBuffer(BaseCompare):
    """In general, sharing of non-flat buffer isn't supported

    The current packing algorithms in StorageRewrite assume a flat
    memory space, and do not support packing of N-d buffers.  For
    buffers with axis separators, normal buffer sharing should be
    disabled.

    Like TestRewriteInPlaceUseOfNonFlatBuffer, except that B and C do
    not have matching shapes.
    """

    def before(A: T.Buffer((16, 16), "float32"), D: T.Buffer((16, 16), "float32")):
        B_data = T.allocate(
            [16, 16],
            dtype="float32",
            scope="global",
        )
        B = T.Buffer(
            [16, 16],
            dtype="float32",
            axis_separators=[1],
            data=B_data,
        )
        C_data = T.allocate(
            [20, 20],
            dtype="float32",
            scope="global",
        )
        C = T.Buffer(
            [20, 20],
            dtype="float32",
            axis_separators=[1],
            data=C_data,
        )

        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

        for i, j in T.grid(16, 16):
            C[i, j] = 2.0 * B[i, j]

        for i, j in T.grid(16, 16):
            D[i, j] = C[i, j]

    expected = before


class TestRewriteDeclBuffer(BaseCompare):
    """A DeclBuffer node may appear in StorageRewrite's input"""

    def before(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
        B = T.decl_buffer(16, dtype="float32")
        C = T.decl_buffer(16, dtype="float32")

        for i in range(16):
            B[i] = A[i]

        for i in range(16):
            C[i] = 2.0 * B[i]

        for i in range(16):
            D[i] = C[i]

    def expected(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
        B = T.decl_buffer(16, dtype="float32")
        C = T.decl_buffer(16, dtype="float32", data=B.data)

        for i in range(16):
            B[i] = A[i]

        for i in range(16):
            C[i] = 2.0 * B[i]

        for i in range(16):
            D[i] = C[i]


class TestNoOrphanedDeclBuffer(BaseCompare):
    """A DeclBuffer of an unused Allocate should be removed

    StorageRewrite removes any allocations that are unused.  When it
    does so, any DeclBuffer that refers to that allocation should also
    be removed.
    """

    def before(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
        B = T.decl_buffer(16, dtype="float32")
        C = T.decl_buffer(16, dtype="float32")
        Unused = T.decl_buffer(16, dtype="float32")

        for i in range(16):
            B[i] = A[i]

        for i in range(16):
            C[i] = 2.0 * B[i]

        for i in range(16):
            D[i] = C[i]

    def expected(A: T.Buffer(16, "float32"), D: T.Buffer(16, "float32")):
        B = T.decl_buffer(16, dtype="float32")
        C = T.decl_buffer(16, dtype="float32", data=B.data)

        for i in range(16):
            B[i] = A[i]

        for i in range(16):
            C[i] = 2.0 * B[i]

        for i in range(16):
            D[i] = C[i]


if __name__ == "__main__":
    tvm.testing.main()
