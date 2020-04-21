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

def test_lower_warp_memory_local_scope():
    m = 128
    A = te.placeholder((m,), name='A')
    B = te.compute((m,), lambda i: A[i] + 3, name='B')

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

    cuda_target = tvm.target.create("cuda")
    assert cuda_target.thread_warp_size == 32
    mod = tvm.lower(s, [A, B], name="f")

    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", cuda_target))(mod)
    fdevice = tvm.tir.transform.SplitHostDevice()(mod)["f_kernel0"]
    mod = tvm.IRModule.from_expr(fdevice)
    fdevice = tvm.tir.transform.LowerWarpMemory()(mod)["f_kernel0"]
    assert(fdevice.body.body.value.value == "local")
    assert(fdevice.body.body.body.extents[0].value == 2)

def test_lower_warp_memory_cuda_end_to_end():
    def check_cuda(dtype):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        m = 128
        A = te.placeholder((m,), name='A', dtype=dtype)
        B = te.compute((m,), lambda i: A[i // 32 * 32 + (i + 1) % 32], name='B')

        cuda_target = tvm.target.create("cuda")
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

            ctx = tvm.gpu(0)
            func = tvm.build(s, [A, B], "cuda")
            A_np = np.array(list(range(m)), dtype=dtype)
            B_np = np.array(
                    list(range(1, 32)) + [0] +
                    list(range(33, 64)) + [32] +
                    list(range(65, 96)) + [64] +
                    list(range(97, 128)) + [96],
                    dtype=dtype)
            A_nd = tvm.nd.array(A_np, ctx)
            B_nd = tvm.nd.array(np.zeros(B_np.shape, dtype=B_np.dtype), ctx)
            func(A_nd, B_nd)
            tvm.testing.assert_allclose(B_nd.asnumpy(), B_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")

def test_lower_warp_memory_cuda_half_a_warp():
    def check_cuda(dtype):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        m = 16
        A = te.placeholder((m,), name='A', dtype=dtype)
        B = te.compute((m,), lambda i: A[(i + 1) % m], name='B')

        cuda_target = tvm.target.create("cuda")
        assert cuda_target.thread_warp_size == 2 * m
        with cuda_target:
            s = te.create_schedule(B.op)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")

            AA = s.cache_read(A, "warp", [B])
            xo, xi = s[B].split(B.op.axis[0], nparts=1)
            s[B].bind(xi, tx)
            s[B].bind(xo, bx)
            s[AA].compute_at(s[B], xo)
            xo, xi = s[AA].split(s[AA].op.axis[0], nparts=1)
            s[AA].bind(xo, bx)
            s[AA].bind(xi, tx)

            ctx = tvm.gpu(0)
            func = tvm.build(s, [A, B], "cuda")
            A_np = np.array(list(range(m)), dtype=dtype)
            B_np = np.array(list(range(1, m)) + [0], dtype=dtype)
            A_nd = tvm.nd.array(A_np, ctx)
            B_nd = tvm.nd.array(np.zeros(B_np.shape, dtype=B_np.dtype), ctx)
            func(A_nd, B_nd)
            tvm.testing.assert_allclose(B_nd.asnumpy(), B_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")

def test_lower_warp_memory_cuda_2_buffers():
    def check_cuda(dtype):
        if not tvm.gpu(0).exist or not tvm.runtime.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        m = 32
        A = te.placeholder((m,), name='A', dtype=dtype)
        B = te.placeholder((m,), name='B', dtype=dtype)
        C = te.compute((m,), lambda i: A[(i + 1) % m] + B[(i + 1) % m], name='C')

        cuda_target = tvm.target.create("cuda")
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

            ctx = tvm.gpu(0)
            func = tvm.build(s, [A, B, C], "cuda")
            AB_np = np.array(list(range(m)), dtype=dtype)
            C_np = np.array(list(range(1, m)) + [0], dtype=dtype) * 2
            A_nd = tvm.nd.array(AB_np, ctx)
            B_nd = tvm.nd.array(AB_np, ctx)
            C_nd = tvm.nd.array(np.zeros(C_np.shape, dtype=C_np.dtype), ctx)
            func(A_nd, B_nd, C_nd)
            tvm.testing.assert_allclose(C_nd.asnumpy(), C_np, rtol=1e-3)

    check_cuda("float32")
    check_cuda("float16")

if __name__ == "__main__":
    test_lower_warp_memory_local_scope()
    test_lower_warp_memory_cuda_end_to_end()
    test_lower_warp_memory_cuda_half_a_warp()
    test_lower_warp_memory_cuda_2_buffers()
