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
import numpy as np
import tvm.testing


def test_for():
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 10, name="j") as j:
            A[j] = A[j] + 2

    body = ib.get()
    assert isinstance(body, tvm.tir.AttrStmt)
    body = body.body
    assert isinstance(body, tvm.tir.Allocate)
    body = body.body
    assert isinstance(body, tvm.tir.For)
    body = body.body
    assert isinstance(body, tvm.tir.SeqStmt)
    assert isinstance(body[1], tvm.tir.For)


def test_if():
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    A = ib.pointer("float32", name="A")
    tmod = tvm.tir.truncmod
    with ib.for_range(0, n, name="i") as i:
        with ib.if_scope(tmod(i, 2) == 0):
            A[i] = A[i] + 1
        with ib.else_scope():
            A[0] = A[i] + 2

    body = ib.get()
    assert A == A
    assert isinstance(body, tvm.tir.For)
    body = body.body
    assert isinstance(body, tvm.tir.IfThenElse)
    assert isinstance(body.condition, tvm.tir.EQ)
    assert isinstance(body.then_case.index, tvm.tir.Var)
    assert body.else_case.index.value == 0


def test_prefetch():
    A = tvm.tir.decl_buffer((10, 20), name="A")
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")

    with ib.for_range(0, n, name="i") as i:
        ib.emit(
            tvm.tir.Prefetch(
                A, [tvm.ir.Range.from_min_extent(i + 1, 2), tvm.ir.Range.from_min_extent(0, 20)]
            )
        )
    body = ib.get()
    assert body.body.bounds[0].extent.value == 2


def test_cpu():
    n = 1024
    dtype = "float32"
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    def test_device_ir(A, B, C):
        n = A.shape[0]
        max_threads = 8
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        with ib.for_range(0, n, name="i") as i:
            Cptr[i] = Aptr[i] + Bptr[i]
        body = ib.get()
        return body

    C = te.extern(
        A.shape,
        [A, B],
        lambda ins, outs: test_device_ir(ins[0], ins[1], outs[0]),
        name="vector_add",
        dtype=dtype,
    )
    s = te.create_schedule(C.op)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return
        # build and invoke the kernel.
        fadd = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

    check_target("llvm")


@tvm.testing.requires_gpu
def test_gpu():
    n = te.size_var("n")
    dtype = "float32"
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    idxd = tvm.tir.indexdiv

    def test_device_ir(A, B, C):
        n = A.shape[0]
        max_threads = 32
        ib = tvm.tir.ir_builder.create()
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", idxd(n + max_threads - 1, max_threads))
        ib.scope_attr(tx, "thread_extent", max_threads)
        idx = bx.var * max_threads + tx.var
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        with ib.if_scope(ib.likely(idx < n)):
            Cptr[idx] = Aptr[idx] + Bptr[idx]
        body = ib.get()
        return body

    C = te.extern(
        A.shape,
        [A, B],
        lambda ins, outs: test_device_ir(ins[0], ins[1], outs[0]),
        name="vector_add",
        dtype=dtype,
    )
    s = te.create_schedule(C.op)
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    def check_target(target):
        n = 1024
        if not tvm.testing.device_enabled(target):
            return
        # build and invoke the kernel.
        fadd = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

    check_target("opencl")
    check_target("cuda")


def test_binary_search():
    def binary_search(ib, n, i, Aptr, Bptr, Cptr):
        lo = ib.allocate("int32", (1,), name="lo", scope="local")
        hi = ib.allocate("int32", (1,), name="hi", scope="local")

        lo[0] = 0
        hi[0] = n
        v = Bptr[i]
        num_loop = int(np.log2(n)) + 1

        with ib.for_range(0, num_loop, test=(lo[0] < hi[0])) as _:
            mid = lo[0] + tvm.tir.floordiv(hi[0] - lo[0], 2).astype("int32")
            with ib.if_scope(Aptr[mid] < v):
                lo[0] = mid + 1
            with ib.else_scope():
                hi[0] = mid

        Cptr[i] = lo[0]

    def searchsorted_ir_cpu(A, B, C, n):
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)

        with ib.for_range(0, n, name="i", kind="parallel") as i:
            binary_search(ib, n, i, Aptr, Bptr, Cptr)

        body = ib.get()

        return body

    def searchsorted_ir_gpu(A, B, C, n):
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)

        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        max_threads = 32
        ib.scope_attr(bx, "thread_extent", tvm.tir.indexdiv(n + max_threads - 1, max_threads))
        ib.scope_attr(tx, "thread_extent", max_threads)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < n):
            binary_search(ib, n, tid, Aptr, Bptr, Cptr)

        body = ib.get()

        return body

    n = 1024
    dtype = "float32"
    A = te.placeholder((n,), name="A", dtype=dtype)
    B = te.placeholder((n,), name="B", dtype=dtype)

    def check_target(target, ir):
        if not tvm.testing.device_enabled(target):
            return

        C = te.extern(
            A.shape,
            [A, B],
            lambda ins, outs: ir(ins[0], ins[1], outs[0], n),
            name="searchsorted_ir",
            dtype="int32",
        )
        s = te.create_schedule(C.op)

        with tvm.transform.PassContext(opt_level=3, disabled_pass=["HoistIfThenElse"]):
            func = tvm.build(s, [A, B, C], target)

        ctx = tvm.context(target, 0)
        a_np = np.random.uniform(size=n).astype(A.dtype)
        b_np = np.random.uniform(size=n).astype(B.dtype)
        a_np = np.sort(a_np)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        func(a, b, c)
        ref = np.searchsorted(a_np, b_np)
        tvm.testing.assert_allclose(c.asnumpy(), ref)

    check_target("llvm", searchsorted_ir_cpu)
    check_target("cuda", searchsorted_ir_gpu)
    check_target("nvptx", searchsorted_ir_gpu)


if __name__ == "__main__":
    test_prefetch()
    test_if()
    test_for()
    test_cpu()
    test_gpu()
    test_binary_search()
