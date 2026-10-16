# Licensed to the Apache Software Foundation (ASF) under one
# ruff: noqa: E501, E741, F841

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

import tvm_tirx
import tvm_tirx.support.nvcc
import tvm_tirx.testing
from tvm_tirx.backend.config import merge_backend_configs
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support.nvcc import have_bf16, have_fp16, have_int8
from tvm_tirx.testing import env


@pytest.fixture(params=["nvcc", "nvrtc"])
def backend_config(request):
    if request.param == "nvrtc":
        pytest.importorskip("cuda.bindings.nvrtc")
    return {"cuda": {"compiler": request.param}}


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_vectorize_add(*, backend_config):
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "float16" and not have_fp16(tvm_tirx.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        if dtype == "int8" and not have_int8(tvm_tirx.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        vec_dtype = f"{dtype}x{lanes}"
        one = tvm_tirx.tirx.const(1, vec_dtype)
        num_blocks = (n + num_thread - 1) // num_thread

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n,), vec_dtype), B: T.Tensor((n,), vec_dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", num_blocks) as i_0:
                    with T.launch_thread("threadIdx.x", num_thread) as i_1:
                        if i_0 * num_thread + i_1 < n:
                            B[i_0 * num_thread + i_1] = A[i_0 * num_thread + i_1] + one

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.empty((n,), vec_dtype, dev).copyfrom(
                np.random.uniform(size=(n, lanes))
            )
            c = tvm_tirx.runtime.empty((n,), vec_dtype, dev)
            fun(a, c)
            tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + 1)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda("float32", 64, 2)
    check_cuda("float32", 64, 3)
    check_cuda("float32", 64, 4)
    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("uint8", 64, 2)
    check_cuda("uint8", 64, 3)
    check_cuda("uint8", 64, 4)
    check_cuda("float16", 64, 2)
    check_cuda("float16", 64, 4)
    check_cuda("float16", 64, 6)
    check_cuda("float16", 64, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_bf16_vectorize_add(*, backend_config):
    if not have_bf16(tvm_tirx.cuda(0).compute_version):
        print("skip because gpu does not support bf16")
        return
    num_thread = 8

    def np_float2np_bf16(arr):
        """Convert a numpy array of float to a numpy array
        of bf16 in uint16"""
        orig = arr.view("<u4")
        bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
        return np.right_shift(orig + bias, 16).astype("uint16")

    def np_bf162np_float(arr):
        """Convert a numpy array of bf16 (uint16) to a numpy array
        of float"""
        u32 = np.left_shift(arr.astype("uint32"), 16)
        return u32.view("<f4")

    def check_cuda(n, lanes):
        vec_dtype = f"bfloat16x{lanes}"
        num_blocks = n // num_thread
        one = tvm_tirx.ir.prim.vector_expr.Broadcast(tvm_tirx.tirx.const(1, "bfloat16"), lanes)

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n,), vec_dtype), B: T.Tensor((n,), vec_dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", num_blocks) as i_0:
                    with T.launch_thread("threadIdx.x", num_thread) as i_1:
                        B[i_0 * num_thread + i_1] = A[i_0 * num_thread + i_1] + one

        with tvm_tirx.ir.transform.PassContext(
            disabled_pass=["tirx.BF16Promote", "tirx.BF16CastElimination", "tirx.BF16TypeLowering"]
        ):
            fun = tvm_tirx.tirx.compile(
                Module, target={"kind": "cuda", "backend_config": backend_config}
            )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            np_a = np.random.uniform(size=(n, lanes)).astype("float32")
            np_a = np_bf162np_float(np_float2np_bf16(np_a))
            a = tvm_tirx.runtime.empty((n,), vec_dtype, dev).copyfrom(np_float2np_bf16(np_a))
            c = tvm_tirx.runtime.empty((n,), vec_dtype, dev)
            fun(a, c)
            c = tvm_tirx.runtime.empty((n, lanes), "uint16", dev).copyfrom(c)
            tvm_tirx.testing.assert_allclose(c.numpy(), np_float2np_bf16(np_a + 1))

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda(64, 2)
    check_cuda(64, 4)
    check_cuda(64, 6)
    check_cuda(64, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_multiply_add(*, backend_config):
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "int8" and not have_int8(tvm_tirx.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        vec_dtype = f"{dtype}x{lanes}"
        num_blocks = n // num_thread

        @I.ir_module
        class Module:
            @T.function
            def main(
                A: T.Tensor((n,), vec_dtype),
                B: T.Tensor((n,), vec_dtype),
                C: T.Tensor((n,), "int32"),
                D: T.Tensor((n,), "int32"),
            ):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", num_blocks) as i_0:
                    with T.launch_thread("threadIdx.x", num_thread) as i_1:
                        D[i_0 * num_thread + i_1] = T.call_pure_extern(
                            "__dp4a",
                            A[i_0 * num_thread + i_1],
                            B[i_0 * num_thread + i_1],
                            C[i_0 * num_thread + i_1],
                            ty="int32",
                        )

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_b = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_c = np.random.randint(low=0, high=127, size=(n,))
        np_d = [sum(x * y) + z for x, y, z in zip(np_a, np_b, np_c)]

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.empty((n,), vec_dtype, dev).copyfrom(np_a)
            b = tvm_tirx.runtime.empty((n,), vec_dtype, dev).copyfrom(np_b)
            c = tvm_tirx.runtime.empty((n,), "int32", dev).copyfrom(np_c)
            d = tvm_tirx.runtime.empty((n,), "int32", dev)
            fun(a, b, c, d)
            tvm_tirx.testing.assert_allclose(d.numpy(), np_d)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda("int8", 64, 4)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_vectorize_load(*, backend_config):
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        vec_dtype = f"{dtype}x{lanes}"
        num_blocks = n // num_thread

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n,), vec_dtype), B: T.Tensor((n,), vec_dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", num_blocks) as i_0:
                    with T.launch_thread("threadIdx.x", num_thread) as i_1:
                        B[i_0 * num_thread + i_1] = A[i_0 * num_thread + i_1]

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
            a = tvm_tirx.runtime.empty((n,), vec_dtype, dev).copyfrom(np_a)
            b = tvm_tirx.runtime.empty((n,), vec_dtype, dev)
            fun(a, b)
            tvm_tirx.testing.assert_allclose(a.numpy(), b.numpy())

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_make_int8(*, backend_config):
    def check_cuda(n, value, lanes):
        dtype = "int8"
        const_value = tvm_tirx.tirx.const(value, ty=dtype)

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n, lanes), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", n) as i:
                    for j in T.vectorized(lanes):
                        A[i, j] = const_value

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            np_a = np.full((n, lanes), value, dtype=dtype)
            a = tvm_tirx.runtime.empty(np_a.shape, dtype, dev)
            fun(a)
            np.testing.assert_equal(a.numpy(), np_a)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda(64, np.uint8(0xAB).view(np.int8), 4)
    check_cuda(64, 0, 4)
    check_cuda(64, -3, 4)
    check_cuda(64, np.uint8(0xAB).view(np.int8), 3)
    check_cuda(64, 0, 3)
    check_cuda(64, -3, 3)
    check_cuda(64, np.uint8(0xAB).view(np.int8), 2)
    check_cuda(64, 0, 2)
    check_cuda(64, -3, 2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_inf_nan(*, backend_config):
    def check_inf_nan(n, value, dtype):
        inf_value = tvm_tirx.tirx.const(value, ty=dtype)

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", 1) as i_0:
                    with T.launch_thread("threadIdx.x", 8) as i_1:
                        if i_0 * 8 + i_1 < n:
                            C[i_0 * 8 + i_1] = inf_value

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.empty((n,), dtype, dev)
            c = tvm_tirx.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_inf_nan(1, -float("inf"), "float32")
    check_inf_nan(1, -float("inf"), "float64")
    check_inf_nan(1, float("inf"), "float32")
    check_inf_nan(1, float("inf"), "float64")
    check_inf_nan(1, float("nan"), "float32")
    check_inf_nan(1, float("nan"), "float64")


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
    ],
)
def test_crossthread_reduction1(target, *, backend_config):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    def sched(nthd):
        n = T.dynamic("n", "int32")
        m = T.dynamic("m", "int32")

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n, m), "float32"), B: T.Tensor((n,), "float32")):
                T.func_attr({"tirx.noalias": True})

                with T.launch_thread("blockIdx.x", n) as i:
                    with T.launch_thread("threadIdx.x", nthd) as m_0:
                        partial = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                        reduced = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                        partial[0] = T.float32(0)
                        for m_1 in range((m + nthd - 1) // nthd):
                            if m_0 * ((m + nthd - 1) // nthd) + m_1 < m:
                                partial[0] = partial[0] + A[i, m_0 * ((m + nthd - 1) // nthd) + m_1]
                        T.gpu_thread_allreduce(
                            T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                            (T.float32(0),),
                            (partial[0],),
                            True,
                            (reduced[0],),
                            (m_0,),
                        )
                        if m_0 == 0:
                            B[i] = reduced[0]

        fun = tvm_tirx.tirx.compile(
            Module,
            target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config}
            if target == "cuda"
            else target,
        )
        return fun

    def verify(nthd):
        func = sched(nthd)
        nn = 3
        # checks three typical cases
        vals = [nthd - 1, nthd, nthd + 1]

        def run_and_check():
            dev = tvm_tirx.device_from_target(target)
            for kk in vals:
                size = (nn, kk)
                a = tvm_tirx.runtime.tensor(np.random.uniform(size=size).astype("float32"), dev)
                b = tvm_tirx.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
                func(a, b)
                tvm_tirx.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    verify(16)
    verify(32)
    verify(64)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
    ],
)
def test_crossthread_reduction2(target, *, backend_config):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    def sched(nthdx, nthdy):
        n = T.dynamic("n", "int32")
        k0 = T.dynamic("k0", "int32")
        k1 = T.dynamic("k1", "int32")

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n, k0, k1), "float32"), B: T.Tensor((n,), "float32")):
                T.func_attr({"tirx.noalias": True})

                with T.launch_thread("blockIdx.x", n) as i:
                    with T.launch_thread("threadIdx.x", nthdx) as k0_0:
                        with T.launch_thread("threadIdx.y", nthdy) as k1_0:
                            partial = T.alloc_tensor(
                                ty_args=[T.Tensor((1,), "float32", scope="local")]
                            )
                            reduced = T.alloc_tensor(
                                ty_args=[T.Tensor((1,), "float32", scope="local")]
                            )
                            partial[0] = T.float32(0)
                            for k0_1, k1_1 in T.grid(
                                (k0 + nthdx - 1) // nthdx, (k1 + nthdy - 1) // nthdy
                            ):
                                if (
                                    k0_0 * ((k0 + nthdx - 1) // nthdx) + k0_1 < k0
                                    and k1_0 * ((k1 + nthdy - 1) // nthdy) + k1_1 < k1
                                ):
                                    partial[0] = (
                                        partial[0]
                                        + A[
                                            i,
                                            k0_0 * ((k0 + nthdx - 1) // nthdx) + k0_1,
                                            k1_0 * ((k1 + nthdy - 1) // nthdy) + k1_1,
                                        ]
                                    )
                            T.gpu_thread_allreduce(
                                T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                                (T.float32(0),),
                                (partial[0],),
                                True,
                                (reduced[0],),
                                (k0_0, k1_0),
                            )
                            if k0_0 == 0 and k1_0 == 0:
                                B[i] = reduced[0]

        func = tvm_tirx.tirx.compile(
            Module,
            target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config}
            if target == "cuda"
            else target,
        )
        return func

    def verify(nthdx, nthdy):
        func = sched(nthdx, nthdy)
        nn = 3
        # checks three typical cases
        vx = [nthdx - 1, nthdx, nthdx + 1]
        vy = [nthdy - 1, nthdy, nthdy + 1]

        def run_and_check():
            dev = tvm_tirx.device_from_target(target)
            for kk0, kk1 in [(x, y) for x in vx for y in vy]:
                size = (nn, kk0, kk1)
                a = tvm_tirx.runtime.tensor(np.random.uniform(size=size).astype("float32"), dev)
                b = tvm_tirx.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
                func(a, b)
                tvm_tirx.testing.assert_allclose(
                    b.numpy(), np.sum(a.numpy(), axis=(1, 2)), rtol=1e-3
                )

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    verify(16, 16)
    verify(32, 32)
    verify(16, 32)
    verify(32, 16)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_reduction_binding(*, backend_config):
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((96, 32), "float32"), B: T.Tensor((96,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for k in range(32):
                with T.launch_thread("blockIdx.x", 3) as m_0:
                    for m_1 in range(32):
                        if k == 0:
                            B[m_0 * 32 + m_1] = T.float32(0.0)
                        B[m_0 * 32 + m_1] = B[m_0 * 32 + m_1] + A[m_0 * 32 + m_1, k]

    func = tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_const_float_to_half(*, backend_config):
    # This import is required to use nvcc to perform code gen;
    # otherwise it is found that the code gen is done by nvrtc.

    half_const = tvm_tirx.tirx.const(0.5, ty="float16")

    @I.ir_module
    class Module:
        @T.function
        def main(a: T.Tensor((2, 3, 4), "float16"), C: T.Tensor((2, 3, 4), "bool")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1) as i_j_k_fused_0:
                with T.launch_thread("threadIdx.x", 64) as i_j_k_fused_1:
                    if i_j_k_fused_0 * 64 + i_j_k_fused_1 < 24:
                        C[
                            (i_j_k_fused_0 * 64 + i_j_k_fused_1) // 12,
                            (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 12 // 4,
                            (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 4,
                        ] = (
                            half_const
                            < a[
                                (i_j_k_fused_0 * 64 + i_j_k_fused_1) // 12,
                                (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 12 // 4,
                                (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 4,
                            ]
                        )

    func = tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})

    shape = (2, 3, 4)
    a_np = np.random.uniform(size=shape).astype("float16")
    c_np = np.zeros(shape=shape, dtype="bool")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        c = tvm_tirx.runtime.tensor(c_np, dev)
        func(a, c)
        np.testing.assert_equal(c.numpy(), a_np > 0.5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_floordiv_with_vectorization(*, backend_config):
    with tvm_tirx.target.Target("cuda"):
        # B[i] = A[floordiv(i, k)]
        n = 256
        k = 37

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", 1) as i_0:
                    with T.launch_thread("threadIdx.x", 64) as i_1_0:
                        for i_1_1 in T.vectorized(4):
                            B[i_0 * 256 + i_1_0 * 4 + i_1_1] = A[
                                (i_0 * 256 + i_1_0 * 4 + i_1_1) // 37
                            ]

        func = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a_np = np.random.uniform(size=(n,)).astype("float32")
            b_np = np.array([a_np[i // k] for i in range(0, n)])
            a_nd = tvm_tirx.runtime.tensor(a_np, dev)
            b_nd = tvm_tirx.runtime.tensor(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
            func(a_nd, b_nd)
            tvm_tirx.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_floormod_with_vectorization(*, backend_config):
    with tvm_tirx.target.Target("cuda"):
        # B[i] = A[floormod(i, k)]
        n = 256
        k = 37

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", 1) as i_0:
                    with T.launch_thread("threadIdx.x", 64) as i_1_0:
                        for i_1_1 in T.vectorized(4):
                            B[i_0 * 256 + i_1_0 * 4 + i_1_1] = A[
                                (i_0 * 256 + i_1_0 * 4 + i_1_1) % 37
                            ]

        func = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a_np = np.random.uniform(size=(n,)).astype("float32")
            b_np = np.array([a_np[i % k] for i in range(0, n)])
            a_nd = tvm_tirx.runtime.tensor(a_np, dev)
            b_nd = tvm_tirx.runtime.tensor(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
            func(a_nd, b_nd)
            tvm_tirx.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


_VECTORIZED_CAST_TYPES_4 = [
    "float16",
    "float32",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "float64",
    "int64",
    "uint64",
]
_VECTORIZED_CAST_TYPES_8 = [
    "float16",
    "float32",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
]


def _skip_vectorized_cast(t0, t1):
    if t0 == t1:
        return True
    # CUDA does support cast between {u}int8 and fp16.
    skip_set = {"float16", "uint8", "int8"}
    return t0 in skip_set and t1 in skip_set


_VECTORIZED_CAST_CASES = [
    (t0, t1, 4)
    for t0 in _VECTORIZED_CAST_TYPES_4
    for t1 in _VECTORIZED_CAST_TYPES_4
    if not _skip_vectorized_cast(t0, t1)
] + [
    (t0, t1, 8)
    for t0 in _VECTORIZED_CAST_TYPES_8
    for t1 in _VECTORIZED_CAST_TYPES_8
    if not _skip_vectorized_cast(t0, t1)
]
_VECTORIZED_CAST_CASES += [("int8", "uint8", 16), ("uint8", "int8", 16)]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("t0,t1,factor", _VECTORIZED_CAST_CASES)
def test_vectorized_casts(t0, t1, factor, *, backend_config):
    if (t0 == "float16" or t1 == "float16") and not have_fp16(tvm_tirx.cuda(0).compute_version):
        print("Skip because gpu does not have fp16 support")
        return

    n = 128
    num_thread = n // factor

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), t0), B: T.Tensor((n,), t1), C: T.Tensor((n,), t0)):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("threadIdx.x", num_thread) as i_0:
                for i_1 in T.vectorized(factor):
                    C[i_0 * factor + i_1] = A[i_0 * factor + i_1] + T.Cast(
                        B[i_0 * factor + i_1], t0
                    )

    func = tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})

    # correctness
    def run_and_check():
        dev = tvm_tirx.cuda(0)
        low, high = (0, 20) if t0.startswith("u") or t1.startswith("u") else (-10, 10)
        a_np = np.random.randint(low, high, size=n).astype(t0)
        b_np = np.random.randint(low, high, size=n).astype(t1)
        c_np = (a_np + b_np).astype(t0)
        a_nd = tvm_tirx.runtime.tensor(a_np, dev)
        b_nd = tvm_tirx.runtime.tensor(b_np, dev)
        c_nd = tvm_tirx.runtime.tensor(np.zeros(c_np.shape, dtype=c_np.dtype), dev)
        func(a_nd, b_nd, c_nd)
        tvm_tirx.testing.assert_allclose(c_nd.numpy(), c_np, rtol=1e-3)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def sched(compute_fn, dtype, n=128, *, backend_config=None):
    """Create a vectorized CUDA module with the given compute function.

    The schedule structure is: split [1, None] -> split [32, None] -> split [None, 4]
    then vectorize innermost, bind blockIdx.x and threadIdx.x.
    For n=128 this gives: blockIdx.x=1, threadIdx.x=32, serial=1, vectorized=4.
    """

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", 1) as i0_0:
                with T.launch_thread("threadIdx.x", 32) as i0_1_0:
                    for i0_1_1_0 in range(1):
                        for i0_1_1_1 in T.vectorized(4):
                            B[i0_1_0 * 4 + i0_1_1_0 * 4 + i0_1_1_1] = compute_fn(
                                A[i0_1_0 * 4 + i0_1_1_0 * 4 + i0_1_1_1]
                            )

    return tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_vectorized_intrin1(*, backend_config):
    test_funcs = [
        (tvm_tirx.tirx.floor, lambda x: np.floor(x)),
        (tvm_tirx.tirx.ceil, lambda x: np.ceil(x)),
        (tvm_tirx.tirx.trunc, lambda x: np.trunc(x)),
        (tvm_tirx.tirx.abs, lambda x: np.fabs(x)),
        (tvm_tirx.tirx.round, lambda x: np.round(x)),
        (tvm_tirx.tirx.exp, lambda x: np.exp(x)),
        (tvm_tirx.tirx.exp2, lambda x: np.exp2(x)),
        (tvm_tirx.tirx.exp10, lambda x: np.power(10, x)),
        (tvm_tirx.tirx.log, lambda x: np.log(x)),
        (tvm_tirx.tirx.log2, lambda x: np.log2(x)),
        (tvm_tirx.tirx.log10, lambda x: np.log10(x)),
        (tvm_tirx.tirx.tan, lambda x: np.tan(x)),
        (tvm_tirx.tirx.cos, lambda x: np.cos(x)),
        (tvm_tirx.tirx.cosh, lambda x: np.cosh(x)),
        (tvm_tirx.tirx.sin, lambda x: np.sin(x)),
        (tvm_tirx.tirx.sinh, lambda x: np.sinh(x)),
        (tvm_tirx.tirx.atan, lambda x: np.arctan(x)),
        (tvm_tirx.tirx.tanh, lambda x: np.tanh(x)),
        (tvm_tirx.tirx.sqrt, lambda x: np.sqrt(x)),
    ]

    def run_test(tvm_intrin, np_func, dtype):
        if dtype == "float16" and not have_fp16(tvm_tirx.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        # set of intrinsics does not support fp16 yet.
        skip_set = {
            tvm_tirx.tirx.abs,
            tvm_tirx.tirx.round,
            tvm_tirx.tirx.tan,
            tvm_tirx.tirx.atan,
            tvm_tirx.tirx.tanh,
            tvm_tirx.tirx.cosh,
            tvm_tirx.tirx.sinh,
        }
        if dtype == "float16" and tvm_intrin in skip_set:
            print(f"Skip because '{tvm_intrin.__name__}' does not support fp16 yet")
            return

        n = 128
        f = sched(tvm_intrin, dtype, n, backend_config=backend_config)

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.tensor(np.random.uniform(0, 1, size=n).astype(dtype), dev)
            b = tvm_tirx.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
            f(a, b)
            tvm_tirx.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    for func in test_funcs:
        run_test(*func, "float32")
        run_test(*func, "float16")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_round_ties_to_even(*, backend_config):
    # The CUDA rule lowers tirx.round to nearbyint/nearbyintf (ties-to-even,
    # matching constant-folding semantics), so exact midpoints must round to
    # even. np.random-based tests never produce midpoints; this pins the tie
    # rule on both the nvcc and nvrtc compile paths (via the autouse fixture).
    # sched() assumes 128 lanes, so the midpoints are tiled to fill it.
    midpoints = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype="float32")
    ties_to_even = np.array([0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, -4.0], dtype="float32")
    test_values = np.tile(midpoints, 16)  # 128 lanes, matches sched()
    expected = np.tile(ties_to_even, 16)

    f = sched(tvm_tirx.tirx.round, "float32", backend_config=backend_config)
    dev = tvm_tirx.cuda(0)
    a = tvm_tirx.runtime.tensor(test_values, dev)
    b = tvm_tirx.runtime.tensor(np.zeros(len(test_values), dtype="float32"), dev)

    def run_and_check():
        f(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), expected, rtol=0, atol=0)
        # every midpoint lane hit an exact tie: pin the sign too (round(-0.5)
        # must produce -0.0, not +0.0)
        assert np.signbit(b.numpy()[4::8]).all(), "negative midpoints must round to -0.0"

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_vectorized_intrin2(dtype="float32", *, backend_config):
    c2 = tvm_tirx.tirx.const(2, ty=dtype)
    test_funcs = [
        (tvm_tirx.tirx.power, lambda x: np.power(x, 2.0)),
        (tvm_tirx.tirx.fmod, lambda x: np.fmod(x, 2.0)),
    ]

    def run_test(tvm_intrin, np_func):
        n = 128
        f = sched(lambda x: tvm_intrin(x, c2), dtype, n, backend_config=backend_config)

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.tensor(np.random.uniform(0, 1, size=n).astype(dtype), dev)
            b = tvm_tirx.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
            f(a, b)
            tvm_tirx.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    for func in test_funcs:
        run_test(*func)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_vectorized_popcount(*, backend_config):
    def ref_popcount(x):
        cnt = 0
        while x:
            x -= x & -x
            cnt += 1
        return cnt

    def run_test(dtype):
        n = 128
        f = sched(lambda x: tvm_tirx.tirx.popcount(x), dtype, n, backend_config=backend_config)

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.tensor(np.random.randint(0, 100000, size=n).astype(dtype), dev)
            b = tvm_tirx.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
            f(a, b)
            ref = np.vectorize(ref_popcount)(a.numpy())
            tvm_tirx.testing.assert_allclose(b.numpy(), ref)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    run_test("uint32")
    run_test("uint64")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_vectorize_load_permute_pad(*, backend_config):
    def check_cuda(dtype, n, l, padding, lanes):
        if dtype == "float16" and not have_fp16(tvm_tirx.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        zero = tvm_tirx.tirx.const(0, dtype)
        dim0 = n // lanes
        dim1 = l + 2 * padding

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n, l), dtype), B: T.Tensor((dim0, dim1, lanes), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", dim0) as i:
                    with T.launch_thread("threadIdx.x", dim1) as j:
                        for k in T.vectorized(lanes):
                            B[i, j, k] = T.if_then_else(
                                j < padding or l + padding <= j, zero, A[i * lanes + k, j - padding]
                            )

        fun = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        )

        np_a = np.random.randint(low=-128, high=127, size=(n, l)).astype(dtype)
        np_a_reshape = np_a.reshape(n // lanes, lanes, l).transpose(0, 2, 1)
        ref = np.pad(
            np_a_reshape, ((0, 0), (padding, padding), (0, 0)), mode="constant", constant_values=0
        )

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            a = tvm_tirx.runtime.empty((n, l), dtype, dev).copyfrom(np_a)
            b = tvm_tirx.runtime.empty((dim0, dim1, lanes), dtype, dev)
            fun(a, b)
            tvm_tirx.testing.assert_allclose(b.numpy(), ref)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    check_cuda("int8", 64, 16, 3, 2)
    check_cuda("uint8", 64, 16, 3, 2)
    check_cuda("int8", 64, 16, 3, 4)
    check_cuda("uint8", 64, 16, 3, 4)
    check_cuda("int32", 64, 16, 3, 4)
    check_cuda("float16", 64, 16, 3, 4)
    check_cuda("float32", 64, 16, 3, 4)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_try_unaligned_vector_load(*, backend_config):
    def build(N, C_N, offset):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((N,), "float16"), C: T.Tensor((C_N,), "float16")):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", C_N // 2) as i_0:
                    for i_1 in T.vectorized(2):
                        C[i_0 * 2 + i_1] = A[i_0 * 2 + i_1 + offset]

        f = tvm_tirx.tirx.compile(
            Module, target={"kind": "cuda", "backend_config": backend_config}
        ).mod

        kernel_source = f.imports[0].inspect_source()
        a_data = np.arange(0, N).astype("float16")

        def run_and_check():
            dev = tvm_tirx.cuda()
            a = tvm_tirx.runtime.tensor(a_data, dev)
            c = tvm_tirx.runtime.tensor(np.zeros(C_N, dtype="float16"), dev)
            f(a, c)
            return c.numpy()

        c = tvm_tirx.testing.run_with_gpu_lock(run_and_check)
        return a_data, c, kernel_source

    # Unaligned case: N=3, C_N=2, offset=1
    a_data, c, kernel_source = build(3, 2, 1)
    # (uint1*)(A + (1)) is invalid
    assert "A_ptr + (1)" not in kernel_source

    expected = a_data[1 : 2 + 1]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"

    # Aligned case: N=4, C_N=2, offset=2
    a_data, c, kernel_source = build(4, 2, 2)
    # (uint1*)(A + (2)) is a valid vector load
    assert "A_ptr + 2" in kernel_source

    expected = a_data[2 : 2 + 2]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_thread_sync_inside_condition(*, backend_config):
    @T.function
    def func2(A: T.Tensor((4, 4), "float32")) -> None:
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32", scope="shared")])
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                if T.gpu_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    T.gpu_storage_sync("shared")
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    @T.function
    def func3(A: T.Tensor((4, 4), "float32")) -> None:
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32", scope="shared")])
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                while T.gpu_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    T.gpu_storage_sync("shared")
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    for func in [func2, func3]:
        mod = tvm_tirx.IRModule({"main": func})
        compiled = tvm_tirx.tirx.compile(
            mod, target={"kind": "cuda", "backend_config": backend_config}
        )
        assert "__syncthreads()" in compiled.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_invalid_reinterpret(*, backend_config):
    @T.function
    def func(A: T.Tensor((4,), "uint32"), B: T.Tensor((4,), "uint8")) -> None:
        with T.launch_thread("threadIdx.x", 4) as tx:
            B[tx] = T.call_intrin("tirx.reinterpret", A[tx], ty="uint8")

    with pytest.raises(RuntimeError):
        tvm_tirx.tirx.compile(func, target={"kind": "cuda", "backend_config": backend_config})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_cuda_tensormap(*, backend_config):
    # fmt: off
    @T.function
    def main(A: T.Tensor((16, 16), dtype='float32', align=16)):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapInit", A_map, "float32", 2, A.data,
                      16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        with T.launch_thread("blockIdx.x", 1) as blockIdx:
            with T.launch_thread("threadIdx.x", 128) as threadIdx:
                if threadIdx == 0:
                    A[0, 0] = T.Cast(T.address_of(A_map), "float32")
    # fmt: on

    mod = tvm_tirx.IRModule({"main": main})
    mod = tvm_tirx.tirx.compile(mod, target={"kind": "cuda", "backend_config": backend_config})
    assert (
        """
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A_ptr, const __grid_constant__ CUtensorMap A_map) {
  if (((int)threadIdx.x) == 0) {
    A_ptr[0] = ((float)((unsigned long long)(&(A_map))));
  }
}""".strip()
        in mod.mod.imports[0].inspect_source()
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_device_func_call(*, backend_config):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def add(a: T.float32, b: T.float32) -> T.float32:
            return a + b

        @T.function
        def main(
            A: T.Tensor((1024, 1024), "float32"),
            B: T.Tensor((1024, 1024), "float32"),
            C: T.Tensor((1024, 1024), "float32"),
        ):
            with T.launch_thread("blockIdx.x", 1024) as bx:
                with T.launch_thread("threadIdx.x", 1024) as tx:
                    C[bx, tx] = Module.add(A[bx, tx], B[bx, tx])

    lib = tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})
    cuda_code = lib.mod.imports[0].inspect_source()
    assert 'extern "C" __device__ float add(float a, float b) {\n  return (a + b);\n}' in cuda_code


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_float_const_hex_format(*, backend_config):
    """Test that float constants are emitted in hexadecimal format for precision"""

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((1024, 1024), "float32"),
        ):
            with T.launch_thread("blockIdx.x", 1024) as bx:
                with T.launch_thread("threadIdx.x", 1024) as tx:
                    A[bx, tx] = T.float32(1 / 27)

    lib = tvm_tirx.tirx.compile(Module, target={"kind": "cuda", "backend_config": backend_config})
    cuda_code = lib.mod.imports[0].inspect_source()
    assert "0x1.2f684bda12f68p-5f" in cuda_code


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_device_host_call_same_func(*, backend_config):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            return a + b

        @T.function
        def main(
            A: T.Tensor((128, 128), "int32"),
            B: T.Tensor((128, 128), "int32"),
            C: T.Tensor((128, 128), "int32"),
        ):
            length: T.let[T.int32] = Module.add(64, 64)  # Call from host
            with T.launch_thread("blockIdx.x", length) as bx:
                with T.launch_thread("threadIdx.x", length) as tx:
                    C[bx, tx] = Module.add(A[bx, tx], B[bx, tx])  # Call from device

    # 1. If we set host to llvm, it will raise an error of
    #    "Return should be transformed to return zero before LLVM code generation."
    #    Need to revisit this.
    # 2. We set a dummy mcpu value for testing purpose,
    #    in order to avoid checking a function is host or device based on the "cpu" substring.
    target = tvm_tirx.target.Target({"kind": "cuda", "mcpu": "dummy_mcpu"}, host="c")
    lib = tvm_tirx.tirx.compile(
        Module, target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config}
    )
    cuda_code = lib.mod.imports[0].inspect_source()
    assert 'extern "C" __device__ int add(int a, int b) {\n  return (a + b);\n}' in cuda_code

    # Run a simple test
    a_np = np.random.randint(0, 10, (128, 128), dtype="int32")
    b_np = np.random.randint(0, 10, (128, 128), dtype="int32")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a_tvm = tvm_tirx.runtime.tensor(a_np, device=dev)
        b_tvm = tvm_tirx.runtime.tensor(b_np, device=dev)
        c_tvm = tvm_tirx.runtime.empty((128, 128), dtype="int32", device=dev)
        lib["main"](a_tvm, b_tvm, c_tvm)
        tvm_tirx.testing.assert_allclose(c_tvm.numpy(), a_np + b_np)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_thread_return(monkeypatch, *, backend_config):
    if not env.has_cuda():
        monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
            with T.launch_thread("blockIdx.x", 32) as bx:
                with T.launch_thread("threadIdx.x", 32) as tx:
                    if bx >= 16 or tx >= 16:
                        T.gpu_thread_return()
                    B[bx, tx] = A[bx, tx]

    lib = tvm_tirx.tirx.compile(
        Module,
        target={
            "kind": "cuda",
            "backend_config": merge_backend_configs(backend_config, {"cuda": {"arch": "sm_80"}}),
        },
    )
    cuda_code = lib.mod.imports[0].inspect_source()
    assert "return;" in cuda_code
    assert "return 0;" not in cuda_code


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_loop_step(*, backend_config):
    @T.function
    def cuda_loop_step(
        A: T.Tensor((1024,), "float32"),
        B: T.Tensor((1024,), "float32"),
        C: T.Tensor((1024,), "float32"),
    ):
        # Each thread computes a strided subset of the i loop: start = tx*3, step = 96 (3 * 32 threads)
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 96) as tx:
                for i in T.serial(tx, 1024, step=96):
                    C[i] = A[i] + B[i]

    target = tvm_tirx.target.Target({"kind": "cuda"})
    lib = tvm_tirx.tirx.compile(
        cuda_loop_step,
        target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config},
    )

    cuda_src = lib.mod.imports[0].inspect_source()
    assert "i += 96" in cuda_src
    a_np = np.random.uniform(1, 100, (1024,)).astype("float32")
    b_np = np.random.uniform(1, 100, (1024,)).astype("float32")
    c_np = np.zeros((1024,), dtype="float32")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a_nd = tvm_tirx.runtime.tensor(a_np, dev)
        b_nd = tvm_tirx.runtime.tensor(b_np, dev)
        c_nd = tvm_tirx.runtime.tensor(c_np, dev)
        lib["main"](a_nd, b_nd, c_nd)
        tvm_tirx.testing.assert_allclose(c_nd.numpy(), a_np + b_np)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
