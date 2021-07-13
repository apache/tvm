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
from tvm.contrib.nvcc import have_fp16

import numpy as np
import tvm.testing


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

    cuda_target = tvm.target.Target("cuda")
    assert cuda_target.thread_warp_size == 32
    mod = tvm.lower(s, [A, B], name="f")

    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", cuda_target))(mod)
    fdevice = tvm.tir.transform.SplitHostDevice()(mod)["f_kernel0"]
    mod = tvm.IRModule.from_expr(fdevice)
    fdevice = tvm.tir.transform.LowerWarpMemory()(mod)["f_kernel0"]
    assert fdevice.body.body.value.value == "local"
    assert fdevice.body.body.body.extents[0].value == 2


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

        cuda_target = tvm.target.Target("cuda")
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

        cuda_target = tvm.target.Target("cuda")
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

        cuda_target = tvm.target.Target("cuda")
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


if __name__ == "__main__":
    test_lower_warp_memory_local_scope()
    test_lower_warp_memory_correct_indices()
    test_lower_warp_memory_cuda_end_to_end()
    test_lower_warp_memory_cuda_half_a_warp()
    test_lower_warp_memory_cuda_2_buffers()
    test_lower_warp_memory_roundup()
