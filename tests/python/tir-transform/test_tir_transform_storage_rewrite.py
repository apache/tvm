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
from tvm.driver.build_module import schedule_to_module
from tvm.script import tir as T


def test_storage_share():
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), name="A")
    num_stage = 5
    B = A
    for t in range(num_stage):
        B = te.compute((m, l), lambda i, j: B[i, j] + (t + 1), name="A%d" % t)

    s = te.create_schedule(B.op)
    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1


def register_mem(scope_tb, max_bits):
    # Register mem
    @tvm.register_func("tvm.info.mem.%s" % scope_tb)
    def mem_info_inp_buffer():
        return tvm.ir.make_node(
            "MemoryInfo", unit_bits=16, max_simd_bits=32, max_num_bits=max_bits, head_address=None
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


def test_inplace_rule():
    m = 10
    A = te.placeholder((m,), name="A")
    A0 = te.compute((m,), lambda i: A[i], name="A0")
    A1 = te.compute((m,), lambda i: A[i] + 1, name="A1")
    AA = te.compute((m,), lambda i: A0[i] + A1[i] + A1[0], name="AA")
    B = te.compute((m,), lambda i: AA[i] + 1, name="B")
    s = te.create_schedule(B.op)
    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 2


def test_storage_combine():
    n = 8
    A = te.placeholder((4,), name="A")
    num_stage = 5
    B = A
    stages = []
    for t in range(num_stage):
        B = te.compute((n,), lambda i: B[i] + B[0] + (t + 1), name="A%d" % t)
        stages.append(B)

    s = te.create_schedule(B.op)
    for S in stages[:-1]:
        s[S].set_scope("global:tag")

    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1
            assert n.extents[0].value == 16

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1


def test_storage_combine_with_vectorization():
    n = 1024
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute((n,), lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "global:tag", readers=[C])
    BB = s.cache_read(B, "global:tag", readers=[C])
    CC = s.cache_write(C, "global:tag")
    s[CC].vectorize(s[CC].op.axis[0])
    mod = schedule_to_module(s, [A, B, C])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    mod = tvm.tir.transform.VectorizeLoop()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    stmt = mod["main"].body
    num_alloc = [0]

    def verify(v):
        # find add op
        if (
            isinstance(v, tvm.tir.Add)
            and isinstance(v.a, tvm.tir.BufferLoad)
            and isinstance(v.b, tvm.tir.BufferLoad)
        ):
            lhs_ramp = v.a.indices[0]
            rhs_ramp = v.b.indices[0]
            # these two ramp load should not overlap
            assert lhs_ramp.lanes == n
            assert rhs_ramp.lanes == n
            assert lhs_ramp.base >= rhs_ramp.base + n or rhs_ramp.base >= lhs_ramp.base + n
        elif isinstance(v, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1


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


def test_storage_share_gpu():
    m = te.var("m")
    A = [te.placeholder((m), name="A")]
    num_stage = 5
    for t in range(num_stage):
        A.append(te.compute((m,), lambda i: A[-1][i] + (t + 1), name="A%d_s" % t))
        A.append(te.compute((m,), lambda i: A[-1][i], name="A%d" % t))
    s = te.create_schedule(A[-1].op)
    for t in range(num_stage):
        x = A[2 * t + 2].op.axis[0]
        bx, tx = s[A[2 * t + 2]].split(x, factor=32)
        s[A[2 * t + 2]].bind(bx, te.thread_axis("blockIdx.x"))
        s[A[2 * t + 2]].bind(tx, te.thread_axis("threadIdx.x"))
        s[A[2 * t + 1]].compute_at(s[A[2 * t + 2]], tx)
        s[A[2 * t + 1]].set_scope("shared")

    mod = schedule_to_module(s, [A[0], A[-1]])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    alloc_stats = {"global": 0, "shared": 0}

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            scope = n.buffer_var.type_annotation.storage_scope
            alloc_stats[scope] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert alloc_stats["global"] == 2
    assert alloc_stats["shared"] == num_stage


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


def test_inplace_rule2(scope_tb="local_TB2", max_bits=1024 * 1024 * 1024):
    # Test Buffer
    register_mem(scope_tb, max_bits)
    m = 10
    A = te.placeholder((m,), name="A")
    C = te.placeholder((m,), name="C")
    D = te.placeholder((m,), name="D")
    A0 = te.compute((m,), lambda i: A[i] + C[i], name="A0")
    A1 = te.compute((m,), lambda i: D[i] * D[i], name="A1")
    A2 = te.compute((m,), lambda i: A0[i] + A1[i], name="A2")
    B = te.compute((m,), lambda i: A2[i], name="B")
    s = te.create_schedule(B.op)
    A0L = s.cache_read(A0, scope_tb, [A2])
    A1L = s.cache_read(A1, scope_tb, [A2])
    A2L = s.cache_read(A2, scope_tb, [B])
    mod = schedule_to_module(s, [A, B, C, D])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    # verify only have one allocations.
    # verify inplace folding works
    num_alloc = [0]

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            num_alloc[0] += 1

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 2


def test_exceed_mem():
    max_bits = 639
    # The critical max_num_bits is between 639 and 640
    loc = -1
    try:
        test_inplace_rule2("local_TEM", max_bits)
    except Exception as e:
        estr = str(e)
        loc = estr.find("Allocation exceed bound of memory")
        assert loc != -1


def test_inplace_rule3():
    # Test Buffer
    scope_tb = "local_TB3"
    max_bits = 1024 * 1024 * 1024

    register_mem(scope_tb, max_bits)
    m = 10
    B0 = te.placeholder((m,), name="B0")
    B1 = te.placeholder((m,), name="B1")
    B2 = te.placeholder((m,), name="B2")
    B3 = te.placeholder((m,), name="B3")
    B4 = te.placeholder((m,), name="B4")
    B5 = te.placeholder((m,), name="B5")

    B6 = te.compute((m,), lambda i: B1[i] * B5[i], name="B6")
    B7 = te.compute((m,), lambda i: B2[i] * B4[i], name="B7")
    B8 = te.compute((m,), lambda i: B6[i] - B7[i], name="B8")

    B9 = te.compute((m,), lambda i: B2[i] * B3[i], name="B9")
    B10 = te.compute((m,), lambda i: B0[i] * B5[i], name="B10")
    B11 = te.compute((m,), lambda i: B9[i] - B10[i], name="B11")

    B12 = te.compute((m,), lambda i: B0[i] * B4[i], name="B12")
    B13 = te.compute((m,), lambda i: B1[i] * B3[i], name="B13")
    B14 = te.compute((m,), lambda i: B12[i] - B13[i], name="B14")

    B = te.compute((m,), lambda i: B8[i] * B11[i] + B14[i], name="B")
    s = te.create_schedule(B.op)

    B1L = s.cache_read(B1, scope_tb, [B6, B13])
    B5L = s.cache_read(B5, scope_tb, [B6, B10])
    B2L = s.cache_read(B2, scope_tb, [B7, B9])
    B4L = s.cache_read(B4, scope_tb, [B7, B12])
    B3L = s.cache_read(B3, scope_tb, [B9, B13])
    B0L = s.cache_read(B0, scope_tb, [B10, B12])

    B8L = s.cache_write(B8, scope_tb)
    B11L = s.cache_write(B11, scope_tb)
    B14L = s.cache_write(B14, scope_tb)
    B6L = s.cache_write(B6, scope_tb)
    B7L = s.cache_write(B7, scope_tb)
    B9L = s.cache_write(B9, scope_tb)
    B10L = s.cache_write(B10, scope_tb)
    B12L = s.cache_write(B12, scope_tb)
    B13L = s.cache_write(B13, scope_tb)

    s[B12].compute_inline()
    s[B13].compute_inline()
    s[B8].compute_inline()
    s[B11].compute_inline()
    s[B14].compute_inline()
    s[B6].compute_inline()
    s[B7].compute_inline()
    s[B9].compute_inline()
    s[B10].compute_inline()

    s = s.normalize()
    mod = schedule_to_module(s, [B0, B1, B2, B3, B4, B5, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.StorageRewrite()(mod)
    stmt = mod["main"].body

    # verify only have one allocations.
    # verify inplace folding works
    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            assert n.extents[0].value == 70

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)


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


def test_replace_dataflow():
    shape = (255,)
    A = te.placeholder(shape, name="A")
    B = te.compute(shape, lambda i: A[i] + A[i], name="B")
    C = te.compute(shape, lambda i: A[i] + B[i], name="C")
    D = te.compute(shape, lambda i: A[i] + C[i], name="D")
    E = te.compute(shape, lambda i: A[i] + D[i], name="E")

    s = te.create_schedule(E.op)
    s.cache_read(A, "local", [B, C, D, E])
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)


def test_large_input():
    @te.hybrid.script
    def compute(a, b):
        n = 16384
        c = output_tensor((n, n), "int32")
        for i in range(n):
            for j in range(n):
                c[i, j] = a[i, j] - b[i, j]
        return c

    n = 16384
    shape = (n, n)
    a = te.placeholder(shape, name="a", dtype="int32")
    b = te.placeholder(shape, name="b", dtype="int32")
    c = te.compute(shape, lambda i, j: compute(a, b)[i, j])
    c = te.compute(shape, lambda i, j: 1 + c[i, j])
    s = te.create_schedule(c.op)
    stmt = tvm.lower(s, [a, b, c])["main"].body

    def verify(n):
        if isinstance(n, tvm.tir.Allocate):
            assert n.extents[0].value == 268435456

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)


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


def test_vulkan_smem_reuse():
    target = tvm.target.Target(
        {
            "keys": ["vulkan", "gpu"],
            "kind": "vulkan",
            "max_num_threads": 256,
            "max_threads_per_block": 256,
            "supports_float32": T.bool(True),
            "supports_int32": T.bool(True),
            "tag": "",
            "thread_warp_size": 1,
        }
    )

    @T.prim_func(private=True)
    def func(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        A_shared = T.allocate([4], "float32", "shared")
        A_local = T.allocate([4], "float32", "local")
        B_shared = T.allocate([4], "float16", "shared")
        A_shared_1 = T.Buffer((4,), data=A_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_1 = T.Buffer((4,), data=A.data)
            A_shared_1[threadIdx_x] = A_1[threadIdx_x]
        A_local_1 = T.Buffer((4,), data=A_local, scope="local")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_local_1[threadIdx_x] = A_shared_1[threadIdx_x]
        B_shared_1 = T.Buffer((4,), "float16", data=B_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            B_shared_1[threadIdx_x] = T.Cast("float16", A_local_1[threadIdx_x])
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        B_1 = T.Buffer((4,), "float16", data=B.data)
        B_1[threadIdx_x] = B_shared_1[threadIdx_x]

    @T.prim_func(private=True)
    def normal_lowering(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        A_shared = T.allocate([4], "float32", "shared")
        A_local = T.allocate([4], "float32", "local")
        A_shared_1 = T.Buffer((4,), data=A_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_1 = T.Buffer((4,), data=A.data)
            A_shared_1[threadIdx_x] = A_1[threadIdx_x]
        A_local_1 = T.Buffer((4,), data=A_local, scope="local")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_local_1[threadIdx_x] = A_shared_1[threadIdx_x]
        A_shared_2 = T.Buffer((4,), "float16", data=A_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_shared_2[threadIdx_x] = T.Cast("float16", A_local_1[threadIdx_x])
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        B_1 = T.Buffer((4,), "float16", data=B.data)
        B_1[threadIdx_x] = A_shared_2[threadIdx_x]

    @T.prim_func(private=True)
    def no_reuse_lowering(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float16")):
        T.func_attr({"target": target, "tir.noalias": T.bool(True)})
        A_shared_1 = T.allocate([4], "float32", "shared")
        A_local_1 = T.allocate([4], "float32", "local")
        B_shared_1 = T.allocate([4], "float16", "shared")
        A_shared_1_1 = T.Buffer((4,), data=A_shared_1, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_1 = T.Buffer((4,), data=A.data)
            A_shared_1_1[threadIdx_x] = A_1[threadIdx_x]
        A_local_1_1 = T.Buffer((4,), data=A_local_1, scope="local")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            A_local_1_1[threadIdx_x] = A_shared_1_1[threadIdx_x]
        B_shared_1_1 = T.Buffer((4,), "float16", data=B_shared_1, scope="shared")
        with T.launch_thread("threadIdx.x", 4) as threadIdx_x:
            B_shared_1_1[threadIdx_x] = T.Cast("float16", A_local_1_1[threadIdx_x])
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        B_1 = T.Buffer((4,), "float16", data=B.data)
        B_1[threadIdx_x] = B_shared_1_1[threadIdx_x]

    # Reuse shared memory when lowering without target.
    mod = tvm.IRModule({"main": func})
    tvm.ir.assert_structural_equal(tvm.lower(mod)["main"], normal_lowering)

    # No shared memory reuse when lowering with target Vulkan.
    mod = tvm.tir.transform.BindTarget(target)(mod)
    tvm.ir.assert_structural_equal(tvm.lower(mod)["main"], no_reuse_lowering)


if __name__ == "__main__":
    tvm.testing.main()
