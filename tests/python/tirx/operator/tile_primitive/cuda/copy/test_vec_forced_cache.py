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
"""Tests for the forced-vec copy cache-semantics config.

Covers the ``vec_256b`` (32-byte / PTX ``.v8``) variant and the
``cache="nc"`` + L1/L2 hint config on the forced-vec copy dispatches:
``Tx.copy(dst, src, dispatch="vec_*", cache="nc", l1_evict=..., ...)``
must lower the global load to ``ld.global.nc`` with the hint suffixes
instead of a plain ``ld.global``.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env


def _build_g2l2g_kernel(n_elements, dtype, dispatch, **copy_config):
    """One thread: A (global) → reg (local) via forced-vec copy (with the
    given config), then reg → B (global) via a plain forced-vec copy."""

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (n_elements,), dtype)
        B = T.match_buffer(B_ptr, (n_elements,), dtype)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        reg = T.alloc_local((n_elements,), dtype)
        Tx.copy(reg[:], A[:], dispatch=dispatch, **copy_config)
        Tx.copy(B[:], reg[:], dispatch=dispatch)

    return kernel


def _build_g2s2g_kernel(n_elements, dtype, dispatch, **copy_config):
    """One thread: A (global) → smem via forced-vec copy (ld + st through a
    local tmp inside the dispatch), then smem → B elementwise."""

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (n_elements,), dtype)
        B = T.match_buffer(B_ptr, (n_elements,), dtype)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((n_elements,), dtype, scope="shared")
        Tx.copy(smem[:], A[:], dispatch=dispatch, **copy_config)
        for i in range(n_elements):
            B[i] = smem[i]

    return kernel


def _compile(kernel):
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
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
    kernel = _build_g2l2g_kernel(8, "int32", "vec_256b")
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
        "vec_128b",
        cache="nc",
        l1_evict="L1::no_allocate",
        prefetch_size="L2::256B",
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
        "vec_256b",
        cache="nc",
        l1_evict="L1::no_allocate",
        l2_evict="L2::evict_first",
        prefetch_size="L2::256B",
    )
    ex, src = _compile(kernel)
    assert "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.u32" in src
    assert "v8" in src
    _run_roundtrip(ex, 8, "int32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vec_128b_nc_global_to_shared():
    kernel = _build_g2s2g_kernel(4, "float32", "vec_128b", cache="nc")
    ex, src = _compile(kernel)
    assert "ld.global.nc.v4.u32" in src
    assert "st.shared.v4.u32" in src
    _run_roundtrip(ex, 4, "float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vec_nc_rejects_non_global_src():
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_local((4,), "float32")
        Tx.copy(reg[:], smem[:], dispatch="vec_128b", cache="nc")
        B[0] = reg[0]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="requires src scope 'global'"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


if __name__ == "__main__":
    tvm.testing.main()
