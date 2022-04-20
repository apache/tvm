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
from tvm.topi.math import cast


def test_for():
    ib = tvm.tir.ir_builder.create()
    n = te.size_var("n")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 10, name="j") as j:
            A[j] = A[j] + 2

    body = ib.get()
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
    assert isinstance(body.then_case.indices[0], tvm.tir.Var)
    assert list(body.else_case.indices) == [0]


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
        dev = tvm.device(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

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
        dev = tvm.device(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_target("opencl")
    check_target("cuda")


def test_while_vectorize():
    """Test while loop + vectorized inner loop"""

    n = 64
    num_iter = 10

    def test_ir(A, B, C):
        ib = tvm.tir.ir_builder.create()
        n = C.shape[0]
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)
        C = ib.buffer_ptr(C)
        i = ib.allocate("int32", (1,), name="i", scope="local")
        i[0] = 0

        with ib.for_range(0, n) as j:
            C[j] = 0.0

        with ib.while_loop(i[0] < num_iter):
            with ib.for_range(0, n, kind="vectorize") as j:
                C[j] += A[j] + B[j]
            i[0] += 1

        return ib.get()

    def check_target(target, ir):
        dtype = "float32"
        A = te.placeholder((n,), name="A", dtype=dtype)
        B = te.placeholder((n,), name="B", dtype=dtype)

        C = te.extern(
            (n,),
            [A, B],
            lambda ins, outs: ir(ins[0], ins[1], outs[0]),
            name="while_vectorize",
            dtype=dtype,
        )
        s = te.create_schedule(C.op)

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(s, [A, B, C], target)

        dev = tvm.device(target, 0)
        a_np = np.random.uniform(size=n).astype(A.dtype)
        b_np = np.random.uniform(size=n).astype(B.dtype)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        func(a, b, c)
        ref = num_iter * (a_np + b_np)
        tvm.testing.assert_allclose(c.numpy(), ref, rtol=1e-5, atol=1e-5)

    check_target("llvm", test_ir)


def test_while_collatz():
    """Test while loop + if"""

    def collatz_ref(n):
        a = n
        i = 0
        while a > 1:
            if a % 2 == 1:
                a = 3 * a + 1
            else:
                a = a >> 1
            i += 1
        return i

    def collatz(ib, n, C):
        i = ib.allocate("int32", (1,), name="i", scope="local")
        a = ib.allocate("int32", (1,), name="a", scope="local")
        i[0] = 0
        a[0] = n
        with ib.while_loop(a[0] > 1):
            with ib.if_scope(tvm.tir.floormod(a[0], 2) == 1):
                a[0] = 3 * a[0] + 1
            with ib.else_scope():
                a[0] = a[0] >> 1
            i[0] += 1

        C[n] = i[0]

    def collatz_ir_cpu(C):
        ib = tvm.tir.ir_builder.create()
        n = C.shape[0]
        C = ib.buffer_ptr(C)

        with ib.for_range(0, n, name="i", kind="parallel") as i:
            collatz(ib, i, C)

        body = ib.get()

        return body

    n = 30

    def check_target(target, ir):
        C = te.extern(
            (n,),
            [],
            lambda ins, outs: ir(outs[0]),
            name="collatz",
            dtype="int32",
        )
        s = te.create_schedule(C.op)

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(s, [C], target)

        dev = tvm.device(target, 0)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        func(c)
        ref = np.array([collatz_ref(i) for i in range(n)])
        tvm.testing.assert_allclose(c.numpy(), ref)

    check_target("llvm", collatz_ir_cpu)


def test_while_mandel():
    n = 160
    shape = (n * 2, n)
    t = 300

    def mandel_ref():
        def complex_sqr(z):
            return np.array([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])

        pixels = np.zeros(shape)

        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                c = np.array([-0.8, np.cos(t) * 0.2])
                z = np.array([i / n - 1, j / n - 0.5]) * 2
                iterations = 0

                while np.linalg.norm(z) < 20 and iterations < 50:
                    z = complex_sqr(z) + c
                    iterations += 1

                pixels[i, j] = 1 - iterations * 0.02

        return pixels

    def mandel(ib, i, j, pixels):
        z = ib.allocate("float32", (2,), name="z", scope="local")
        tmp = ib.allocate("float32", (1,), name="tmp", scope="local")
        iterations = ib.allocate("int32", (1,), name="iterations", scope="local")

        z[0] = (i / float(n) - 1) * 2
        z[1] = (j / float(n) - 0.5) * 2
        iterations[0] = 0
        c = [-0.8, float(np.cos(t)) * 0.2]

        def norm(z):
            return tvm.tir.sqrt(z[0] * z[0] + z[1] * z[1])

        with ib.while_loop(tvm.tir.all(norm(z) < 20, iterations[0] < 50)):
            tmp[0] = z[0]
            z[0] = z[0] * z[0] - z[1] * z[1] + c[0]
            z[1] = z[1] * tmp[0] * 2 + c[1]
            iterations[0] += 1

        pixels[i, j] = 1 - iterations[0] * 0.02

    def mandel_ir_cpu(C):
        ib = tvm.tir.ir_builder.create()
        ny = C.shape[0]
        nx = C.shape[1]
        C = ib.buffer_ptr(C)

        with ib.for_range(0, ny, name="i", kind="parallel") as i:
            with ib.for_range(0, nx, name="j") as j:
                mandel(ib, i, j, C)

        body = ib.get()

        return body

    def mandel_ir_gpu(C):
        ib = tvm.tir.ir_builder.create()
        ny = C.shape[0]
        nx = C.shape[1]
        C = ib.buffer_ptr(C)

        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        by = te.thread_axis("blockIdx.y")
        ty = te.thread_axis("threadIdx.y")

        max_threads = 16
        ib.scope_attr(bx, "thread_extent", tvm.tir.indexdiv(nx + max_threads - 1, max_threads))
        ib.scope_attr(tx, "thread_extent", max_threads)
        ib.scope_attr(by, "thread_extent", tvm.tir.indexdiv(ny + max_threads - 1, max_threads))
        ib.scope_attr(ty, "thread_extent", max_threads)

        tidx = bx * max_threads + tx
        tidy = by * max_threads + ty

        with ib.if_scope(tvm.tir.all(tidx < nx, tidy < ny)):
            mandel(ib, tidy, tidx, C)

        body = ib.get()

        return body

    ref = mandel_ref()

    def check_target(target, ir):
        if not tvm.testing.device_enabled(target):
            return

        C = te.extern(
            shape,
            [],
            lambda ins, outs: ir(outs[0]),
            name="mandel_ir",
            dtype="float32",
        )
        s = te.create_schedule(C.op)

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(s, [C], target)

        dev = tvm.device(target, 0)
        c = tvm.nd.array(np.zeros(shape, dtype=C.dtype), dev)
        func(c)
        tvm.testing.assert_allclose(c.numpy(), ref, rtol=1e-5, atol=1e-5)

    check_target("llvm", mandel_ir_cpu)
    check_target("npvtx", mandel_ir_gpu)
    check_target("cuda", mandel_ir_gpu)
    check_target("vulkan", mandel_ir_gpu)


def test_while_binary_search():
    def binary_search(ib, n, i, Aptr, Bptr, Cptr):
        lo = ib.allocate("int32", (1,), name="lo", scope="local")
        hi = ib.allocate("int32", (1,), name="hi", scope="local")

        lo[0] = 0
        hi[0] = n
        v = Bptr[i]

        with ib.while_loop(lo[0] < hi[0]):
            mid = lo[0] + (hi[0] - lo[0] >> 1)
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

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(s, [A, B, C], target)

        dev = tvm.device(target, 0)
        a_np = np.random.uniform(size=n).astype(A.dtype)
        b_np = np.random.uniform(size=n).astype(B.dtype)
        a_np = np.sort(a_np)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        func(a, b, c)
        ref = np.searchsorted(a_np, b_np)
        tvm.testing.assert_allclose(c.numpy(), ref)

    check_target("llvm", searchsorted_ir_cpu)
    check_target("cuda", searchsorted_ir_gpu)
    check_target("nvptx", searchsorted_ir_gpu)
    check_target("vulkan", searchsorted_ir_gpu)


@tvm.testing.requires_gpu
def test_dyn_shared():
    n = te.size_var("n")
    dtype = "float32"
    A = te.placeholder((n,), name="A")

    def test_device_ir(A, B):
        n = A.shape[0]
        ib = tvm.tir.ir_builder.create()

        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", n)

        temp = ib.allocate(dtype, (n,), scope="shared.dyn")  # n is symbolic size

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)

        temp[tx] = Aptr[tx]
        depth = tvm.tir.log2(cast(n, "float32"))

        with ib.for_range(0, cast(tvm.tir.ceil(depth), n.dtype)) as i:
            ib.emit(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))
            d = n >> (i + 1)
            with ib.if_scope(tx < d):
                temp[tx] += temp[tx + d]

        Bptr[0] = temp[0]
        return ib.get()

    B = te.extern(
        (1,),
        [A],
        lambda ins, outs: test_device_ir(ins[0], outs[0]),
        name="reduce",
        dtype=dtype,
    )
    s = te.create_schedule(B.op)

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            return

        freduce = tvm.build(s, [A, B], target)
        dev = tvm.device(target, 0)

        for n in [512, 1024]:
            a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
            b = tvm.nd.array(np.zeros(1, dtype=B.dtype), dev)
            freduce(a, b)
            tvm.testing.assert_allclose(b.numpy()[0], np.sum(a.numpy()), 1e-4, 1e-4)

    for target in ["cuda", "nvptx"]:
        check_target(target)


if __name__ == "__main__":
    test_prefetch()
    test_if()
    test_for()
    test_cpu()
    test_gpu()
    test_while_vectorize()
    test_while_collatz()
    test_while_mandel()
    test_while_binary_search()
    test_dyn_shared()
