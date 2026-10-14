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
# ruff: noqa: F841
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_rocm_inf_nan():
    def check_inf_nan(n, value, dtype):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", 1) as i_0:
                    with T.launch_thread("threadIdx.x", 128) as i_1:
                        if i_0 * 128 + i_1 < 1:
                            C[i_0 * 128 + i_1] = T.Cast(value, dtype)

        fun = tvm.tirx.compile(Module, "rocm")

        def run_and_check():
            dev = tvm.rocm(0)
            a = tvm.runtime.empty((n,), dtype, dev)
            c = tvm.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_inf_nan(1, -float("inf"), "float32")
    check_inf_nan(1, -float("inf"), "float64")
    check_inf_nan(1, float("inf"), "float32")
    check_inf_nan(1, float("inf"), "float64")
    check_inf_nan(1, float("nan"), "float32")
    check_inf_nan(1, float("nan"), "float64")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_rocm_copy():
    def check_rocm(dtype, n):
        def run_and_check():
            dev = tvm.rocm(0)
            a_np = np.random.uniform(size=(n,)).astype(dtype)
            a = tvm.runtime.empty((n,), dtype, dev).copyfrom(a_np)
            b_np = a.numpy()
            tvm.testing.assert_allclose(a_np, b_np)
            tvm.testing.assert_allclose(a_np, a.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)

    for _ in range(100):
        dtype = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 15)
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_rocm(dtype, int(peturb * (2**logN)))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_rocm_vectorize_add():
    def check_rocm(dtype, n, lanes):
        vec_dtype = f"{dtype}x{lanes}"
        num_blocks = n // 4

        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((n,), vec_dtype), B: T.Tensor((n,), vec_dtype)):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("blockIdx.x", num_blocks) as i_0:
                    with T.launch_thread("threadIdx.x", 4) as i_1:
                        B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + T.Broadcast(T.Cast(1, dtype), lanes)

        fun = tvm.tirx.compile(Module, target="rocm")

        def run_and_check():
            dev = tvm.rocm(0)
            a = tvm.runtime.empty((n,), vec_dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
            c = tvm.runtime.empty((n,), vec_dtype, dev)
            fun(a, c)
            tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_rocm("float32", 64, 2)
    check_rocm("float16", 64, 2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_rocm_warp_shuffle():
    @T.function
    def func(
        A: T.Tensor((32,), dtype="float32"),
    ):
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                A_local = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                mask = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
                t0 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                A_local[0] = A[tx]
                A_local[0] = T.gpu_warp_shuffle(mask[0], A_local[0], 0, 32, 32)
                A[tx] = A_local[0]

    mod = tvm.tirx.compile(func, target="rocm")

    def run_and_check():
        dev = tvm.rocm(0)
        a = tvm.runtime.tensor(np.random.uniform(size=(32,)).astype("float32"), dev)
        mod(a)
        tvm.testing.assert_allclose(a.numpy(), np.ones((32,)) * a.numpy()[0])

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_rocm_vectorized_exp():
    @T.function
    def func(
        A: T.Tensor((4,), dtype="float32"),
        B: T.Tensor((4,), dtype="float32"),
    ):
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 1) as tx:
                for i in T.vectorized(0, 4):
                    B[i] = T.exp2(A[i])

    mod = tvm.tirx.compile(func, target="rocm")

    def run_and_check():
        dev = tvm.rocm(0)
        a = tvm.runtime.tensor(np.ones((4,)).astype("float32"), dev)
        b = tvm.runtime.tensor(np.zeros((4,)).astype("float32"), dev)
        mod(a, b)
        tvm.testing.assert_allclose(b.numpy(), np.exp2(a.numpy()))

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_rocm(), reason="need rocm")
def test_export_load_with_fallback(monkeypatch, tmp_path):
    """Force the codegen wrapper into the fallback branch, then export+load+run."""
    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    host_lib = tvm.tirx.compile(Module, target="rocm")
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    lib_path = str(tmp_path / "lib.so")
    host_lib.export_library(lib_path)
    reloaded = tvm.runtime.load_module(lib_path)

    a_np = np.random.uniform(size=(n,)).astype("float32")
    b_np = np.zeros((n,), dtype="float32")

    def run_and_check():
        dev = tvm.rocm(0)
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        reloaded["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np + 1.0, rtol=1e-5)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
