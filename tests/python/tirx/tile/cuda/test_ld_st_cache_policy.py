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
# pylint: disable=missing-function-docstring
"""Preserve vector widths and cache hints through explicitly selected PTX."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


def _build_g2l2g_kernel(n_elements, dtype, load, store):
    @T.function
    def kernel(A: T.Tensor((n_elements,), dtype), B: T.Tensor((n_elements,), dtype)) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        reg = T.alloc_local((n_elements,), dtype, layout="default")
        T.cuda.tile.apply(T.ptx[load], reg.view("uint32"), A.view("uint32"))
        T.cuda.tile.apply(T.ptx[store], B.view("uint32"), reg.view("uint32"))

    return kernel


def _build_g2s2g_kernel(n_elements, dtype, load, store):
    @T.function
    def kernel(A: T.Tensor((n_elements,), dtype), B: T.Tensor((n_elements,), dtype)) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(
            ty_args=[T.Tensor((n_elements,), dtype, scope="shared", layout="default")]
        )
        reg = T.alloc_local((n_elements,), dtype, layout="default")
        T.cuda.tile.apply(T.ptx[load], reg.view("uint32"), A.view("uint32"))
        T.cuda.tile.apply(T.ptx[store], smem.view("uint32"), reg.view("uint32"))
        for i in range(n_elements):
            B[i] = smem[i]

    return kernel


def _compile(kernel):
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.tirx.compile(mod, target=target, pipeline="tirx")
    return ex, ex.mod.imports[0].inspect_source()


def _run_roundtrip(ex, n_elements, dtype):
    dev = tvm.cuda(0)
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    a_np = np.arange(1, n_elements + 1, dtype=np_dtype) * 3
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(np.zeros((n_elements,), dtype=np_dtype), dev)
    ex(a, b)
    np.testing.assert_array_equal(b.numpy(), a_np)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="PTX .v8 ld/st needs cuda compute >= 10.0")
def test_copy_vec_256b_plain():
    kernel = _build_g2l2g_kernel(8, "int32", "ld.global.v8.u32", "st.global.v8.u32")
    ex, src = _compile(kernel)
    assert "ld.global.v8.u32" in src
    assert "st.global.v8.u32" in src
    assert "ld.global.nc" not in src
    _run_roundtrip(ex, 8, "int32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vec_128b_nc_with_hints():
    # NB: ptxas only accepts an L2 eviction-priority qualifier on 256-bit
    # loads (.v8.b32 / .v4.b64), so the 128-bit case sticks to L1 + prefetch.
    kernel = _build_g2l2g_kernel(
        4,
        "int32",
        "ld.global.nc.L1::no_allocate.L2::256B.v4.u32",
        "st.global.v4.u32",
    )
    ex, src = _compile(kernel)
    assert "ld.global.nc.L1::no_allocate.L2::256B.v4.u32" in src
    assert "st.global.v4.u32" in src
    _run_roundtrip(ex, 4, "int32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="PTX .v8 ld/st needs cuda compute >= 10.0")
def test_copy_vec_256b_nc_with_hints():
    kernel = _build_g2l2g_kernel(
        8,
        "int32",
        "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32",
        "st.global.v8.u32",
    )
    ex, src = _compile(kernel)
    assert "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32" in src
    assert "v8" in src
    _run_roundtrip(ex, 8, "int32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vec_128b_nc_global_to_shared():
    kernel = _build_g2s2g_kernel(4, "float32", "ld.global.nc.v4.u32", "st.shared.v4.u32")
    ex, src = _compile(kernel)
    assert "ld.global.nc.v4.u32" in src
    assert "st.shared.v4.u32" in src
    _run_roundtrip(ex, 4, "float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vec_nc_rejects_non_global_src():
    @T.function
    def kernel(B: T.Tensor((4,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", layout="default")])
        reg = T.alloc_local((4,), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.nc.v4.u32"], reg.view("uint32"), smem.view("uint32"))
        B[0] = reg[0]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="expected storage scope.*global"):
            tvm.tirx.compile(mod, target=target, pipeline="tirx")


if __name__ == "__main__":
    tvm.testing.main()
