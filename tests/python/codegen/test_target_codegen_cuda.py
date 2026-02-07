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
import tvm.contrib.nvcc
import tvm.testing
from tvm.contrib.nvcc import have_bf16, have_fp16, have_int8
from tvm.script import ir as I
from tvm.script import tir as T


@pytest.fixture(autouse=True, params=["nvcc", "nvrtc"])
def setup_cuda_compile_mode(request):
    mode = request.param
    if mode == "nvrtc":
        try:
            from cuda.bindings import nvrtc
        except ImportError:
            pytest.skip("cuda-python not available, skipping nvrtc tests")

    orig_func = tvm.contrib.nvcc.tvm_callback_cuda_compile

    def compile_mode_wrapper(code, target):
        if mode == "nvcc":
            return tvm.contrib.nvcc.compile_cuda(code, target_format="fatbin", compiler="nvcc")
        elif mode == "nvrtc":
            return tvm.contrib.nvcc.compile_cuda(code, target_format="cubin", compiler="nvrtc")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    tvm.register_global_func("tvm_callback_cuda_compile", compile_mode_wrapper, override=True)
    # yield back to the original function so that each test runs twice
    yield
    tvm.register_global_func("tvm_callback_cuda_compile", orig_func, override=True)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_add():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        if dtype == "int8" and not have_int8(tvm.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        vec_dtype = "%sx%d" % (dtype, lanes)
        one = tvm.tir.const(1, vec_dtype)
        num_blocks = (n + num_thread - 1) // num_thread

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n,), vec_dtype), B: T.Buffer((n,), vec_dtype)):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(num_thread, thread="threadIdx.x"):
                        with T.sblock("B"):
                            v_i = T.axis.spatial(n, i_0 * num_thread + i_1)
                            T.where(i_0 * num_thread + i_1 < n)
                            T.reads(A[v_i])
                            T.writes(B[v_i])
                            B[v_i] = A[v_i] + one

        fun = tvm.compile(Module, target="cuda")

        dev = tvm.cuda(0)
        a = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.runtime.empty((n,), vec_dtype, dev)
        fun(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

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


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_bf16_vectorize_add():
    if not have_bf16(tvm.cuda(0).compute_version):
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
        vec_dtype = "bfloat16x%d" % lanes
        num_blocks = n // num_thread
        one = tvm.tir.Broadcast(tvm.tir.const(1, "bfloat16"), lanes)

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n,), vec_dtype), B: T.Buffer((n,), vec_dtype)):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(num_thread, thread="threadIdx.x"):
                        with T.sblock("B"):
                            v_i = T.axis.spatial(n, i_0 * num_thread + i_1)
                            T.reads(A[v_i])
                            T.writes(B[v_i])
                            B[v_i] = A[v_i] + one

        with tvm.transform.PassContext(
            disabled_pass=["tir.BF16Promote", "tir.BF16CastElimination", "tir.BF16TypeLowering"]
        ):
            fun = tvm.compile(Module, target="cuda")
        dev = tvm.cuda(0)
        np_a = np.random.uniform(size=(n, lanes)).astype("float32")
        np_a = np_bf162np_float(np_float2np_bf16(np_a))
        a = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np_float2np_bf16(np_a))
        c = tvm.runtime.empty((n,), vec_dtype, dev)
        fun(a, c)
        c = tvm.runtime.empty((n, lanes), "uint16", dev).copyfrom(c)
        tvm.testing.assert_allclose(c.numpy(), np_float2np_bf16(np_a + 1))

    check_cuda(64, 2)
    check_cuda(64, 4)
    check_cuda(64, 6)
    check_cuda(64, 8)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_multiply_add():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "int8" and not have_int8(tvm.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        vec_dtype = "%sx%d" % (dtype, lanes)
        num_blocks = n // num_thread

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((n,), vec_dtype),
                B: T.Buffer((n,), vec_dtype),
                C: T.Buffer((n,), "int32"),
                D: T.Buffer((n,), "int32"),
            ):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(num_thread, thread="threadIdx.x"):
                        with T.sblock("D"):
                            v_i = T.axis.spatial(n, i_0 * num_thread + i_1)
                            T.reads(A[v_i], B[v_i], C[v_i])
                            T.writes(D[v_i])
                            D[v_i] = T.call_pure_extern("int32", "__dp4a", A[v_i], B[v_i], C[v_i])

        fun = tvm.compile(Module, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_b = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_c = np.random.randint(low=0, high=127, size=(n,))
        np_d = [sum(x * y) + z for x, y, z in zip(np_a, np_b, np_c)]
        dev = tvm.cuda(0)
        a = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np_a)
        b = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np_b)
        c = tvm.runtime.empty((n,), "int32", dev).copyfrom(np_c)
        d = tvm.runtime.empty((n,), "int32", dev)
        fun(a, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), np_d)

    check_cuda("int8", 64, 4)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_load():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        dev = tvm.cuda(0)
        vec_dtype = "%sx%d" % (dtype, lanes)
        num_blocks = n // num_thread

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n,), vec_dtype), B: T.Buffer((n,), vec_dtype)):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(num_thread, thread="threadIdx.x"):
                        with T.sblock("B"):
                            v_i = T.axis.spatial(n, i_0 * num_thread + i_1)
                            T.reads(A[v_i])
                            T.writes(B[v_i])
                            B[v_i] = A[v_i]

        fun = tvm.compile(Module, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        a = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np_a)
        b = tvm.runtime.empty((n,), vec_dtype, dev)
        fun(a, b)
        tvm.testing.assert_allclose(a.numpy(), b.numpy())

    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_make_int8():
    def check_cuda(n, value, lanes):
        dtype = "int8"
        dev = tvm.cuda(0)
        const_value = tvm.tir.const(value, dtype=dtype)

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n, lanes), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(n, thread="blockIdx.x"):
                    for j in T.vectorized(lanes):
                        with T.sblock("A"):
                            v_i, v_j = T.axis.remap("SS", [i, j])
                            T.reads()
                            T.writes(A[v_i, v_j])
                            A[v_i, v_j] = const_value

        fun = tvm.compile(Module, target="cuda")

        np_a = np.full((n, lanes), value, dtype=dtype)
        a = tvm.runtime.empty(np_a.shape, dtype, dev)
        fun(a)
        np.testing.assert_equal(a.numpy(), np_a)

    check_cuda(64, np.uint8(0xAB).view(np.int8), 4)
    check_cuda(64, 0, 4)
    check_cuda(64, -3, 4)
    check_cuda(64, np.uint8(0xAB).view(np.int8), 3)
    check_cuda(64, 0, 3)
    check_cuda(64, -3, 3)
    check_cuda(64, np.uint8(0xAB).view(np.int8), 2)
    check_cuda(64, 0, 2)
    check_cuda(64, -3, 2)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_inf_nan():
    target = "cuda"

    def check_inf_nan(dev, n, value, dtype):
        inf_value = tvm.tir.const(value, dtype=dtype)

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n,), dtype), C: T.Buffer((n,), dtype)):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(1, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(8, thread="threadIdx.x"):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(n, i_0 * 8 + i_1)
                            T.where(i_0 * 8 + i_1 < n)
                            T.reads()
                            T.writes(C[v_i])
                            C[v_i] = inf_value

        fun = tvm.compile(Module, target="cuda")

        a = tvm.runtime.empty((n,), dtype, dev)
        c = tvm.runtime.empty((n,), dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_crossthread_reduction1(target, dev):
    def sched(nthd):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(var_A: T.handle, var_B: T.handle):
                T.func_attr({"tir.noalias": True})
                n, m = T.int32(), T.int32()
                A = T.match_buffer(var_A, (n, m))
                B = T.match_buffer(var_B, (n,))
                for i in T.thread_binding(n, thread="blockIdx.x"):
                    for m_0 in T.thread_binding(nthd, thread="threadIdx.x"):
                        for m_1 in range((m + nthd - 1) // nthd):
                            with T.sblock("B"):
                                v_i = T.axis.spatial(n, i)
                                v_m = T.axis.reduce(m, m_0 * ((m + nthd - 1) // nthd) + m_1)
                                T.where(m_0 * ((m + nthd - 1) // nthd) + m_1 < m)
                                T.reads(A[v_i, v_m])
                                T.writes(B[v_i])
                                with T.init():
                                    B[v_i] = T.float32(0.0)
                                B[v_i] = B[v_i] + A[v_i, v_m]

        fun = tvm.compile(Module, target="cuda")
        return fun

    def verify(nthd):
        func = sched(nthd)
        nn = 3
        # checks three typical cases
        vals = [nthd - 1, nthd, nthd + 1]
        for kk in [x for x in vals]:
            size = (nn, kk)
            a = tvm.runtime.tensor(np.random.uniform(size=size).astype("float32"), dev)
            b = tvm.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
            func(a, b)
            tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-3)

    verify(16)
    verify(32)
    verify(64)


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_crossthread_reduction2(target, dev):
    def sched(nthdx, nthdy):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(var_A: T.handle, var_B: T.handle):
                T.func_attr({"tir.noalias": True})
                n, k0, k1 = T.int32(), T.int32(), T.int32()
                A = T.match_buffer(var_A, (n, k0, k1))
                B = T.match_buffer(var_B, (n,))
                for i in T.thread_binding(n, thread="blockIdx.x"):
                    for k0_0 in T.thread_binding(nthdx, thread="threadIdx.x"):
                        for k0_1 in range((k0 + nthdx - 1) // nthdx):
                            for k1_0 in T.thread_binding(nthdy, thread="threadIdx.y"):
                                for k1_1 in range((k1 + nthdy - 1) // nthdy):
                                    with T.sblock("B"):
                                        v_i = T.axis.spatial(n, i)
                                        v_k0 = T.axis.reduce(
                                            k0, k0_0 * ((k0 + nthdx - 1) // nthdx) + k0_1
                                        )
                                        v_k1 = T.axis.reduce(
                                            k1, k1_0 * ((k1 + nthdy - 1) // nthdy) + k1_1
                                        )
                                        T.where(
                                            k0_0 * ((k0 + nthdx - 1) // nthdx) + k0_1 < k0
                                            and k1_0 * ((k1 + nthdy - 1) // nthdy) + k1_1 < k1
                                        )
                                        T.reads(A[v_i, v_k0, v_k1])
                                        T.writes(B[v_i])
                                        with T.init():
                                            B[v_i] = T.float32(0.0)
                                        B[v_i] = B[v_i] + A[v_i, v_k0, v_k1]

        func = tvm.compile(Module, target="cuda")
        return func

    def verify(nthdx, nthdy):
        func = sched(nthdx, nthdy)
        nn = 3
        # checks three typical cases
        vx = [nthdx - 1, nthdx, nthdx + 1]
        vy = [nthdy - 1, nthdy, nthdy + 1]
        for kk0, kk1 in [(x, y) for x in vx for y in vy]:
            size = (nn, kk0, kk1)
            a = tvm.runtime.tensor(np.random.uniform(size=size).astype("float32"), dev)
            b = tvm.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
            func(a, b)
            tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=(1, 2)), rtol=1e-3)

    verify(16, 16)
    verify(32, 32)
    verify(16, 32)
    verify(32, 16)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_reduction_binding():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((96, 32), "float32"), B: T.Buffer((96,), "float32")):
            T.func_attr({"tir.noalias": True})
            for k in range(32):
                for m_0 in T.thread_binding(3, thread="blockIdx.x"):
                    for m_1 in range(32):
                        with T.sblock("B"):
                            v_m = T.axis.spatial(96, m_0 * 32 + m_1)
                            v_k = T.axis.reduce(32, k)
                            T.reads(A[v_m, v_k])
                            T.writes(B[v_m])
                            with T.init():
                                B[v_m] = T.float32(0.0)
                            B[v_m] = B[v_m] + A[v_m, v_k]

    func = tvm.compile(Module, target="cuda")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_const_float_to_half():
    # This import is required to use nvcc to perform code gen;
    # otherwise it is found that the code gen is done by nvrtc.

    half_const = tvm.tir.const(0.5, dtype="float16")

    @I.ir_module
    class Module:
        @T.prim_func
        def main(a: T.Buffer((2, 3, 4), "float16"), C: T.Buffer((2, 3, 4), "bool")):
            T.func_attr({"tir.noalias": True})
            for i_j_k_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for i_j_k_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(2, (i_j_k_fused_0 * 64 + i_j_k_fused_1) // 12)
                        v_j = T.axis.spatial(3, (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 12 // 4)
                        v_k = T.axis.spatial(4, (i_j_k_fused_0 * 64 + i_j_k_fused_1) % 4)
                        T.where(i_j_k_fused_0 * 64 + i_j_k_fused_1 < 24)
                        T.reads(a[v_i, v_j, v_k])
                        T.writes(C[v_i, v_j, v_k])
                        C[v_i, v_j, v_k] = half_const < a[v_i, v_j, v_k]

    func = tvm.compile(Module, target="cuda")

    dev = tvm.cuda(0)
    shape = (2, 3, 4)
    a_np = np.random.uniform(size=shape).astype("float16")
    c_np = np.zeros(shape=shape, dtype="bool")
    a = tvm.runtime.tensor(a_np, dev)
    c = tvm.runtime.tensor(c_np, dev)
    func(a, c)
    np.testing.assert_equal(c.numpy(), a_np > 0.5)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_floordiv_with_vectorization():
    with tvm.target.cuda():
        # B[i] = A[floordiv(i, k)]
        n = 256
        k = 37

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(1, thread="blockIdx.x"):
                    for i_1_0 in T.thread_binding(64, thread="threadIdx.x"):
                        for i_1_1 in T.vectorized(4):
                            with T.sblock("B"):
                                v_i = T.axis.spatial(256, i_0 * 256 + i_1_0 * 4 + i_1_1)
                                T.reads(A[v_i // 37])
                                T.writes(B[v_i])
                                B[v_i] = A[v_i // 37]

        func = tvm.compile(Module, target="cuda")

        dev = tvm.cuda(0)
        a_np = np.random.uniform(size=(n,)).astype("float32")
        b_np = np.array([a_np[i // k] for i in range(0, n)])
        a_nd = tvm.runtime.tensor(a_np, dev)
        b_nd = tvm.runtime.tensor(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
        func(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_floormod_with_vectorization():
    with tvm.target.cuda():
        # B[i] = A[floormod(i, k)]
        n = 256
        k = 37

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(1, thread="blockIdx.x"):
                    for i_1_0 in T.thread_binding(64, thread="threadIdx.x"):
                        for i_1_1 in T.vectorized(4):
                            with T.sblock("B"):
                                v_i = T.axis.spatial(256, i_0 * 256 + i_1_0 * 4 + i_1_1)
                                T.reads(A[v_i % 37])
                                T.writes(B[v_i])
                                B[v_i] = A[v_i % 37]

        func = tvm.compile(Module, target="cuda")

        dev = tvm.cuda(0)
        a_np = np.random.uniform(size=(n,)).astype("float32")
        b_np = np.array([a_np[i % k] for i in range(0, n)])
        a_nd = tvm.runtime.tensor(a_np, dev)
        b_nd = tvm.runtime.tensor(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
        func(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_casts():
    def check(t0, t1, factor):
        if (t0 == "float16" or t1 == "float16") and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        n = 128
        num_thread = n // factor

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n,), t0), B: T.Buffer((n,), t1), C: T.Buffer((n,), t0)):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(num_thread, thread="threadIdx.x"):
                    for i_1 in T.vectorized(factor):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(n, i_0 * factor + i_1)
                            T.reads(A[v_i], B[v_i])
                            T.writes(C[v_i])
                            C[v_i] = A[v_i] + T.Cast(t0, B[v_i])

        func = tvm.compile(Module, target="cuda")

        # correctness
        dev = tvm.cuda(0)
        low, high = (0, 20) if t0.startswith("u") or t1.startswith("u") else (-10, 10)
        a_np = np.random.randint(low, high, size=n).astype(t0)
        b_np = np.random.randint(low, high, size=n).astype(t1)
        c_np = (a_np + b_np).astype(t0)
        a_nd = tvm.runtime.tensor(a_np, dev)
        b_nd = tvm.runtime.tensor(b_np, dev)
        c_nd = tvm.runtime.tensor(np.zeros(c_np.shape, dtype=c_np.dtype), dev)
        func(a_nd, b_nd, c_nd)
        tvm.testing.assert_allclose(c_nd.numpy(), c_np, rtol=1e-3)

    def skip(t0, t1):
        if t0 == t1:
            return True
        # CUDA does support cast between {u}int8 and fp16.
        skip_set = {"float16", "uint8", "int8"}
        if t0 in skip_set and t1 in skip_set:
            return True
        return False

    types_4 = [
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
    types_8 = ["float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32"]
    for t0, t1 in [(x, y) for x in types_4 for y in types_4 if not skip(x, y)]:
        check(t0, t1, 4)
    for t0, t1 in [(x, y) for x in types_8 for y in types_8 if not skip(x, y)]:
        check(t0, t1, 8)
    check("int8", "uint8", 16)
    check("uint8", "int8", 16)


def sched(compute_fn, dtype, n=128):
    """Create a vectorized CUDA module with the given compute function.

    The schedule structure is: split [1, None] -> split [32, None] -> split [None, 4]
    then vectorize innermost, bind blockIdx.x and threadIdx.x.
    For n=128 this gives: blockIdx.x=1, threadIdx.x=32, serial=1, vectorized=4.
    """

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((n,), dtype), B: T.Buffer((n,), dtype)):
            T.func_attr({"tir.noalias": True})
            for i0_0 in T.thread_binding(1, thread="blockIdx.x"):
                for i0_1_0 in T.thread_binding(32, thread="threadIdx.x"):
                    for i0_1_1_0 in range(1):
                        for i0_1_1_1 in T.vectorized(4):
                            with T.sblock("B"):
                                v_i0 = T.axis.spatial(n, i0_1_0 * 4 + i0_1_1_0 * 4 + i0_1_1_1)
                                T.reads(A[v_i0])
                                T.writes(B[v_i0])
                                B[v_i0] = compute_fn(A[v_i0])

    return tvm.compile(Module, target="cuda")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_intrin1():
    test_funcs = [
        (tvm.tir.floor, lambda x: np.floor(x)),
        (tvm.tir.ceil, lambda x: np.ceil(x)),
        (tvm.tir.trunc, lambda x: np.trunc(x)),
        (tvm.tir.abs, lambda x: np.fabs(x)),
        (tvm.tir.round, lambda x: np.round(x)),
        (tvm.tir.exp, lambda x: np.exp(x)),
        (tvm.tir.exp2, lambda x: np.exp2(x)),
        (tvm.tir.exp10, lambda x: np.power(10, x)),
        (tvm.tir.log, lambda x: np.log(x)),
        (tvm.tir.log2, lambda x: np.log2(x)),
        (tvm.tir.log10, lambda x: np.log10(x)),
        (tvm.tir.tan, lambda x: np.tan(x)),
        (tvm.tir.cos, lambda x: np.cos(x)),
        (tvm.tir.cosh, lambda x: np.cosh(x)),
        (tvm.tir.sin, lambda x: np.sin(x)),
        (tvm.tir.sinh, lambda x: np.sinh(x)),
        (tvm.tir.atan, lambda x: np.arctan(x)),
        (tvm.tir.tanh, lambda x: np.tanh(x)),
        (tvm.tir.sqrt, lambda x: np.sqrt(x)),
    ]

    def run_test(tvm_intrin, np_func, dtype):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        # set of intrinsics does not support fp16 yet.
        skip_set = {
            tvm.tir.abs,
            tvm.tir.round,
            tvm.tir.tan,
            tvm.tir.atan,
            tvm.tir.tanh,
            tvm.tir.cosh,
            tvm.tir.sinh,
        }
        if dtype == "float16" and tvm_intrin in skip_set:
            print("Skip because '{0}' does not support fp16 yet".format(tvm_intrin.__name__))
            return

        n = 128
        f = sched(tvm_intrin, dtype, n)
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(np.random.uniform(0, 1, size=n).astype(dtype), dev)
        b = tvm.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func, "float32")
        run_test(*func, "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_intrin2(dtype="float32"):
    c2 = tvm.tir.const(2, dtype=dtype)
    test_funcs = [
        (tvm.tir.power, lambda x: np.power(x, 2.0)),
        (tvm.tir.fmod, lambda x: np.fmod(x, 2.0)),
    ]

    def run_test(tvm_intrin, np_func):
        n = 128
        f = sched(lambda x: tvm_intrin(x, c2), dtype, n)
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(np.random.uniform(0, 1, size=n).astype(dtype), dev)
        b = tvm.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_popcount():
    def ref_popcount(x):
        cnt = 0
        while x:
            x -= x & -x
            cnt += 1
        return cnt

    def run_test(dtype):
        n = 128
        f = sched(lambda x: tvm.tir.popcount(x), dtype, n)
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(np.random.randint(0, 100000, size=n).astype(dtype), dev)
        b = tvm.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
        f(a, b)
        ref = np.vectorize(ref_popcount)(a.numpy())
        tvm.testing.assert_allclose(b.numpy(), ref)

    run_test("uint32")
    run_test("uint64")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_load_permute_pad():
    def check_cuda(dtype, n, l, padding, lanes):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        dev = tvm.cuda(0)
        zero = tvm.tir.const(0, dtype)
        dim0 = n // lanes
        dim1 = l + 2 * padding

        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((n, l), dtype), B: T.Buffer((dim0, dim1, lanes), dtype)):
                T.func_attr({"tir.noalias": True})
                for i in T.thread_binding(dim0, thread="blockIdx.x"):
                    for j in T.thread_binding(dim1, thread="threadIdx.x"):
                        for k in T.vectorized(lanes):
                            with T.sblock("B"):
                                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                                T.reads(A[v_i * lanes + v_k, v_j - padding])
                                T.writes(B[v_i, v_j, v_k])
                                B[v_i, v_j, v_k] = T.if_then_else(
                                    v_j < padding or l + padding <= v_j,
                                    zero,
                                    A[v_i * lanes + v_k, v_j - padding],
                                )

        fun = tvm.compile(Module, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, l)).astype(dtype)
        a = tvm.runtime.empty((n, l), dtype, dev).copyfrom(np_a)
        b = tvm.runtime.empty((dim0, dim1, lanes), dtype, dev)
        fun(a, b)
        np_a_reshape = np_a.reshape(n // lanes, lanes, l).transpose(0, 2, 1)
        ref = np.pad(
            np_a_reshape, ((0, 0), (padding, padding), (0, 0)), mode="constant", constant_values=0
        )
        tvm.testing.assert_allclose(b.numpy(), ref)

    check_cuda("int8", 64, 16, 3, 2)
    check_cuda("uint8", 64, 16, 3, 2)
    check_cuda("int8", 64, 16, 3, 4)
    check_cuda("uint8", 64, 16, 3, 4)
    check_cuda("int32", 64, 16, 3, 4)
    check_cuda("float16", 64, 16, 3, 4)
    check_cuda("float32", 64, 16, 3, 4)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_try_unaligned_vector_load():
    def build(N, C_N, offset):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((N,), "float16"), C: T.Buffer((C_N,), "float16")):
                T.func_attr({"tir.noalias": True})
                for i_0 in T.thread_binding(C_N // 2, thread="threadIdx.x"):
                    for i_1 in T.vectorized(2):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(C_N, i_0 * 2 + i_1)
                            T.reads(A[v_i + offset])
                            T.writes(C[v_i])
                            C[v_i] = A[v_i + offset]

        f = tvm.tir.build(Module, target="cuda")

        kernel_source = f.imports[0].inspect_source()
        dev = tvm.cuda()
        a_data = np.arange(0, N).astype("float16")
        a = tvm.runtime.tensor(a_data, dev)
        c = tvm.runtime.tensor(np.zeros(C_N, dtype="float16"), dev)
        f(a, c)

        return a_data, c.numpy(), kernel_source

    # Unaligned case: N=3, C_N=2, offset=1
    a_data, c, kernel_source = build(3, 2, 1)
    # (uint1*)(A + (1)) is invalid
    assert "A + (1)" not in kernel_source

    expected = a_data[1 : 2 + 1]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"

    # Aligned case: N=4, C_N=2, offset=2
    a_data, c, kernel_source = build(4, 2, 2)
    # (uint1*)(A + (2)) is a valid vector load
    assert "A + 2" in kernel_source

    expected = a_data[2 : 2 + 2]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_thread_sync_inside_condition():
    @T.prim_func
    def func1(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                if A[0, 0] > 1.0:
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    @T.prim_func
    def func2(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                if T.tvm_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    @T.prim_func
    def func3(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                while T.tvm_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    mod = tvm.IRModule({"main": func1})
    with pytest.raises(tvm.error.InternalError):
        tvm.compile(mod, target="cuda")

    mod = tvm.IRModule({"main": func2})
    tvm.compile(mod, target="cuda")

    mod = tvm.IRModule({"main": func3})
    tvm.compile(mod, target="cuda")


@tvm.testing.requires_cuda
def test_invalid_reinterpret():
    @T.prim_func
    def func(A: T.Buffer((4,), "uint32"), B: T.Buffer((4,), "uint8")) -> None:
        for tx in T.thread_binding(4, "threadIdx.x"):
            B[tx] = T.call_intrin("uint8", "tir.reinterpret", A[tx])

    with pytest.raises(tvm.error.TVMError):
        tvm.compile(func, target="cuda")


@tvm.testing.requires_cuda
@tvm.testing.requires_cuda_compute_version(9)
def test_cuda_tensormap():
    # fmt: off
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapInit", A_map, "float32", 2, A.data,
                      16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                if threadIdx == 0:
                    A[0, 0] = T.reinterpret("float64", A_map)
    # fmt: on

    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target="cuda")
    assert (
        """
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A, const __grid_constant__ CUtensorMap A_map) {
  if (((int)threadIdx.x) == 0) {
    A[0] = ((float)(*(double *)(&(A_map))));
  }
}""".strip()
        in mod.mod.imports[0].inspect_source()
    )


@tvm.testing.requires_cuda
def test_cuda_device_func_call():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def add(a: T.float32, b: T.float32) -> T.float32:
            return a + b

        @T.prim_func
        def main(
            A: T.Buffer((1024, 1024), "float32"),
            B: T.Buffer((1024, 1024), "float32"),
            C: T.Buffer((1024, 1024), "float32"),
        ):
            for bx in T.thread_binding(1024, "blockIdx.x"):
                for tx in T.thread_binding(1024, "threadIdx.x"):
                    C[bx, tx] = Module.add(A[bx, tx], B[bx, tx])

    lib = tvm.compile(Module, target="cuda")
    cuda_code = lib.mod.imports[0].inspect_source()
    assert 'extern "C" __device__ float add(float a, float b) {\n  return (a + b);\n}' in cuda_code


@tvm.testing.requires_cuda
def test_cuda_float_const_hex_format():
    """Test that float constants are emitted in hexadecimal format for precision"""

    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((1024, 1024), "float32"),
        ):
            for bx in T.thread_binding(1024, "blockIdx.x"):
                for tx in T.thread_binding(1024, "threadIdx.x"):
                    A[bx, tx] = T.float32(1 / 27)

    lib = tvm.compile(Module, target="cuda")
    cuda_code = lib.mod.imports[0].inspect_source()
    assert "0x1.2f684bda12f68p-5f" in cuda_code


@tvm.testing.requires_cuda
def test_device_host_call_same_func():
    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def add(a: T.int32, b: T.int32) -> T.int32:
            return a + b

        @T.prim_func
        def main(
            A: T.Buffer((128, 128), "int32"),
            B: T.Buffer((128, 128), "int32"),
            C: T.Buffer((128, 128), "int32"),
        ):
            length: T.int32 = Module.add(64, 64)  # Call from host
            for bx in T.thread_binding(length, "blockIdx.x"):
                for tx in T.thread_binding(length, "threadIdx.x"):
                    C[bx, tx] = Module.add(A[bx, tx], B[bx, tx])  # Call from device

    # 1. If we set host to llvm, it will raise an error of
    #    "the tir.ret should be transformed to return zero before the llvm code generation."
    #    Need to revisit this.
    # 2. We set a dummy mcpu value for testing purpose,
    #    in order to avoid checking a function is host or device based on the "cpu" substring.
    target = tvm.target.Target({"kind": "cuda", "mcpu": "dummy_mcpu"}, host="c")
    lib = tvm.compile(Module, target=target)
    cuda_code = lib.mod.imports[0].inspect_source()
    assert 'extern "C" __device__ int add(int a, int b) {\n  return (a + b);\n}' in cuda_code

    # Run a simple test
    dev = tvm.cuda(0)
    a_np = np.random.randint(0, 10, (128, 128), dtype="int32")
    b_np = np.random.randint(0, 10, (128, 128), dtype="int32")
    a_tvm = tvm.runtime.tensor(a_np, device=dev)
    b_tvm = tvm.runtime.tensor(b_np, device=dev)
    c_tvm = tvm.runtime.empty((128, 128), dtype="int32", device=dev)
    lib["main"](a_tvm, b_tvm, c_tvm)
    tvm.testing.assert_allclose(c_tvm.numpy(), a_np + b_np)


@tvm.testing.requires_cuda
def test_thread_return():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")):
            for bx in T.thread_binding(32, "blockIdx.x"):
                for tx in T.thread_binding(32, "threadIdx.x"):
                    if bx >= 16 or tx >= 16:
                        T.thread_return()
                    B[bx, tx] = A[bx, tx]

    lib = tvm.compile(Module, target="cuda")
    cuda_code = lib.mod.imports[0].inspect_source()
    assert "return;" in cuda_code


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_loop_step():
    @T.prim_func
    def cuda_loop_step(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        # Each thread computes a strided subset of the i loop: start = tx*3, step = 96 (3 * 32 threads)
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(96, "threadIdx.x"):
                for i in T.serial(tx, 1024, step=96):
                    C[i] = A[i] + B[i]

    target = tvm.target.Target({"kind": "cuda"})
    with tvm.transform.PassContext(disabled_pass=["s_tir.CanonicalizeLoop"]):
        lib = tvm.compile(cuda_loop_step, target=target)

    cuda_src = lib.mod.imports[0].inspect_source()
    assert "i += 96" in cuda_src
    dev = tvm.cuda(0)
    a_np = np.random.uniform(1, 100, (1024,)).astype("float32")
    b_np = np.random.uniform(1, 100, (1024,)).astype("float32")
    c_np = np.zeros((1024,), dtype="float32")
    a_nd = tvm.runtime.tensor(a_np, dev)
    b_nd = tvm.runtime.tensor(b_np, dev)
    c_nd = tvm.runtime.tensor(c_np, dev)
    lib["main"](a_nd, b_nd, c_nd)
    tvm.testing.assert_allclose(c_nd.numpy(), a_np + b_np)


if __name__ == "__main__":
    tvm.testing.main()
