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
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import te, tir
from tvm.contrib.nvcc import have_fp16


def _run_passes(mod):
    cuda_target = tvm.target.Target("cuda", host="llvm")
    assert cuda_target.thread_warp_size == 32
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", cuda_target))(mod)
    mod = tvm.tir.transform.AnnotateDeviceRegions()(mod)
    mod = tvm.tir.transform.SplitHostDevice()(mod)
    mod = tvm.tir.transform.LowerWarpMemory()(mod)
    return mod


@tvm.testing.requires_cuda
def test_lower_warp_memory_local_scope():
    m = 128
    A = te.placeholder((m,), name="A")
    B = te.compute((m,), lambda i: A[i] + 3, name="B")

    s = te.create_schedule(B.op)
    AA = s.cache_read(A, "warp", [B])
    xo, xi = s[B].split(B.op.axis[0], 64)
    xi0, xi1 = s[B].split(xi, factor=32)
    tx = te.thread_axis("threadIdx.x")
    s[B].bind(xi1, tx)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[AA].compute_at(s[B], xo)
    xo, xi = s[AA].split(s[AA].op.axis[0], 32)
    s[AA].bind(xi, tx)

    # lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        mod = tvm.lower(s, [A, B], name="f")

    mod = _run_passes(mod)
    fdevice = mod["f_kernel"]

    allocate = fdevice
    while not isinstance(allocate, tir.Allocate):
        allocate = allocate.body

    assert allocate.buffer_var.type_annotation.storage_scope == "local"
    assert allocate.extents[0].value == 2


@tvm.testing.requires_cuda
def test_lower_warp_memory_correct_indices():
    n = 32
    A = te.placeholder((2, n, n), name="A", dtype="float32")
    C = te.compute((2, n, n), lambda x, i, j: A(x, i, (j + 1) % n), name="C")

    s = te.create_schedule(C.op)
    bk_x = te.thread_axis("blockIdx.x")
    th_y = te.thread_axis("threadIdx.y")
    th_x = te.thread_axis("threadIdx.x")
    B = s.cache_read(A, "warp", [C])
    cx, ci, cj = C.op.axis
    bx, bi, bj = B.op.axis
    s[C].bind(cj, th_x)
    s[C].bind(cx, bk_x)
    s[B].compute_at(s[C], cx)
    s[B].bind(bi, th_y)
    s[B].bind(bj, th_x)

    bounds = tvm.te.schedule.InferBound(s)
    ir = tvm.te.schedule.ScheduleOps(s, bounds)
    inner_func = ir.body.body.body
    store_A_warp = inner_func.seq[0].body.body
    indices = list(store_A_warp.indices)

    # A.warp is actually many buffers, one for each warp, although they are all called A.warp
    # 1. If we are accessing from different threads within a same warp (different
    #    threadIdx.x), we need to distinguish between each elements using threadIdx.x,
    #    so threadIdx.x is one if the indices.
    # 2. If we are accessing from different warps (different threadIdx.y), we are actually
    #    assessing different buffers, so there is no need to distinguish from elements,
    #    and therefore threadIdx.y is NOT a index.
    idx_names = map(lambda x: x.name, filter(lambda x: type(x) is tvm.tir.expr.Var, indices))
    assert "threadIdx.x" in idx_names
    assert "threadIdx.y" not in idx_names


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_lower_warp_memory_cuda_end_to_end():
    def check_cuda(dtype):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        m = 128
        A = te.placeholder((m,), name="A", dtype=dtype)
        B = te.compute((m,), lambda i: A[i // 32 * 32 + (i + 1) % 32], name="B")

        cuda_target = tvm.target.Target("cuda", host="llvm")
        assert cuda_target.thread_warp_size == 32
        with cuda_target:
            s = te.create_schedule(B.op)
            AA = s.cache_read(A, "warp", [B])
            xo, xi = s[B].split(B.op.axis[0], 64)
            xi0, xi1 = s[B].split(xi, factor=32)
            tx = te.thread_axis("threadIdx.x")
            s[B].bind(xi1, tx)
            s[B].bind(xo, te.thread_axis("blockIdx.x"))
            s[AA].compute_at(s[B], xo)
            xo, xi = s[AA].split(s[AA].op.axis[0], 32)
            s[AA].bind(xi, tx)

            dev = tvm.cuda(0)
            # building with the CSE pass disabled as otherwise it would do some commoning
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
                func = tvm.build(s, [A, B], "cuda")
            A_np = np.array(list(range(m)), dtype=dtype)
            B_np = np.array(
                list(range(1, 32))
                + [0]
                + list(range(33, 64))
                + [32]
                + list(range(65, 96))
                + [64]
                + list(range(97, 128))
                + [96],
                dtype=dtype,
            )
            A_nd = tvm.nd.array(A_np, dev)
            B_nd = tvm.nd.array(np.zeros(B_np.shape, dtype=B_np.dtype), dev)
            func(A_nd, B_nd)
            tvm.testing.assert_allclose(B_nd.numpy(), B_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_lower_warp_memory_cuda_half_a_warp():
    def check_cuda(dtype):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        n, m = 16, 16
        A = te.placeholder(
            (
                n,
                m,
            ),
            name="A",
            dtype=dtype,
        )
        B = te.compute(
            (
                n,
                m,
            ),
            lambda j, i: A[j, (i + 1) % m],
            name="B",
        )

        cuda_target = tvm.target.Target("cuda", host="llvm")
        assert cuda_target.thread_warp_size == 2 * m
        with cuda_target:
            s = te.create_schedule(B.op)
            tx = te.thread_axis("threadIdx.x")
            ty = te.thread_axis("threadIdx.y")
            bx = te.thread_axis("blockIdx.x")

            AA = s.cache_read(A, "warp", [B])
            y, x = B.op.axis
            z, y = s[B].split(y, nparts=2)
            s[B].bind(x, tx)
            s[B].bind(y, ty)
            s[B].bind(z, bx)
            s[AA].compute_at(s[B], y)
            _, x = AA.op.axis
            s[AA].bind(x, tx)

            dev = tvm.cuda(0)
            # building with the CSE pass disabled as otherwise it would do some commoning
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
                func = tvm.build(s, [A, B], "cuda")
            A_np = np.array([list(range(i, m + i)) for i in range(n)], dtype=dtype)
            B_np = np.array([list(range(1 + i, m + i)) + [i] for i in range(n)], dtype=dtype)
            A_nd = tvm.nd.array(A_np, dev)
            B_nd = tvm.nd.array(np.zeros(B_np.shape, dtype=B_np.dtype), dev)
            func(A_nd, B_nd)
            tvm.testing.assert_allclose(B_nd.numpy(), B_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_lower_warp_memory_cuda_2_buffers():
    def check_cuda(dtype):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        m = 32
        A = te.placeholder((m,), name="A", dtype=dtype)
        B = te.placeholder((m,), name="B", dtype=dtype)
        C = te.compute((m,), lambda i: A[(i + 1) % m] + B[(i + 1) % m], name="C")

        cuda_target = tvm.target.Target("cuda", host="llvm")
        assert m <= cuda_target.thread_warp_size
        with cuda_target:
            s = te.create_schedule(C.op)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")

            AA = s.cache_read(A, "warp", [C])
            BB = s.cache_read(B, "warp", [C])
            xo, xi = s[C].split(C.op.axis[0], nparts=1)
            s[C].bind(xi, tx)
            s[C].bind(xo, bx)
            s[AA].compute_at(s[C], xo)
            s[BB].compute_at(s[C], xo)
            xo, xi = s[AA].split(s[AA].op.axis[0], nparts=1)
            s[AA].bind(xo, bx)
            s[AA].bind(xi, tx)
            xo, xi = s[BB].split(s[BB].op.axis[0], nparts=1)
            s[BB].bind(xo, bx)
            s[BB].bind(xi, tx)

            dev = tvm.cuda(0)
            # building with the CSE pass disabled as otherwise it would do some commoning
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
                func = tvm.build(s, [A, B, C], "cuda")
            AB_np = np.array(list(range(m)), dtype=dtype)
            C_np = np.array(list(range(1, m)) + [0], dtype=dtype) * 2
            A_nd = tvm.nd.array(AB_np, dev)
            B_nd = tvm.nd.array(AB_np, dev)
            C_nd = tvm.nd.array(np.zeros(C_np.shape, dtype=C_np.dtype), dev)
            func(A_nd, B_nd, C_nd)
            tvm.testing.assert_allclose(C_nd.numpy(), C_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")


@tvm.testing.requires_gpu
def test_lower_warp_memory_roundup():
    def check(device, m):
        A = te.placeholder((m,), name="A")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")

        with tvm.target.Target(device):
            s = te.create_schedule(B.op)
            xo, xi = s[B].split(B.op.axis[0], factor=32)
            tx = te.thread_axis("threadIdx.x")
            s[B].bind(xo, te.thread_axis("blockIdx.x"))
            s[B].bind(xi, tx)

            AA = s.cache_read(A, "warp", [B])
            _, yi = s[AA].split(s[AA].op.axis[0], factor=32)
            s[AA].bind(yi, tx)
            s[AA].compute_at(s[B], xo)

            dev = tvm.device(device, 0)
            # building with the CSE pass disabled as otherwise it would do some commoning
            with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
                func = tvm.build(s, [A, B], device)
            A_np = np.random.uniform(size=(m,)).astype(A.dtype)
            B_np = np.zeros(shape=(m,)).astype(B.dtype)
            A_nd = tvm.nd.array(A_np, dev)
            B_nd = tvm.nd.array(B_np, dev)
            func(A_nd, B_nd)
            B_np = A_np + 1
            tvm.testing.assert_allclose(B_nd.numpy(), B_np)

    for device in ["cuda", "rocm"]:
        if not tvm.testing.device_enabled(device):
            print("skip because", device, "is not enabled..")
            continue
        check(device, m=31)
        check(device, m=32)
        check(device, m=33)
        check(device, m=63)
        check(device, m=64)
        check(device, m=65)


@tvm.testing.requires_cuda
def test_lower_warp_memory_same_thread():
    m = n = 128
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), name="k")
    B = te.compute((m,), lambda i: te.sum(A[i, k], axis=[k]))

    s = te.create_schedule(B.op)
    BB = s.cache_write(B, "warp")
    tx = te.thread_axis("threadIdx.x")
    xo, xi = s[B].split(B.op.axis[0], factor=32)
    s[B].bind(xi, tx)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[BB].compute_at(s[B], xo)
    xo, xi = s[BB].split(s[BB].op.axis[0], factor=32)
    s[BB].bind(xi, tx)

    # lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        mod = tvm.lower(s, [A, B], name="f")

    mod = _run_passes(mod)
    fdevice = mod["f_kernel"]
    assert "tvm_warp_shuffle" not in fdevice.script()


@tvm.testing.requires_cuda
def test_lower_warp_memory_divide_by_factor():
    ib = tvm.tir.ir_builder.IRBuilder()
    bx = te.thread_axis("blockIdx.x")
    tx = te.thread_axis("threadIdx.x")

    with ib.new_scope():
        ib.scope_attr(bx, "thread_extent", 32)
        ib.scope_attr(tx, "thread_extent", 32)
        t = ib.allocate("float32", 16, name="t", scope="warp")
        n = ib.allocate("float32", 16, name="n", scope="local")
        n[0] = t[0]

    stmt = ib.get()
    func = tvm.tir.PrimFunc([], stmt)
    func = func.with_attr("from_legacy_te_schedule", True)
    # lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        mod = tvm.lower(func, name="f")
    with pytest.raises(tvm.error.TVMError, match="Divide by zero") as cm:
        _run_passes(mod)


if __name__ == "__main__":
    tvm.testing.main()
