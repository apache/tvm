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
# pylint: disable=missing-docstring
"""Numeric test of the RDNA (gfx11, wave32) WMMA tensor intrinsics in tensor_intrin/rocm_wmma.py."""

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir.tensor_intrin.rocm_wmma import (
    ROCM_WMMA_F16F16F32_INTRIN,
    ROCM_WMMA_FILL_16x16_F32_INTRIN,
    ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN,
    ROCM_WMMA_LOAD_16x16_B_SHARED_F16_INTRIN,
    ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN,
    ROCM_WMMA_STORE_16x16_F32_INTRIN,
    shared_16x16_to_local_32x8_layout_C,
    shared_16x16_to_local_32x16_layout_A,
    shared_16x16_to_local_32x16_layout_B,
)
from tvm.s_tir.testing.tir import mfma_schedule

M = N = K = 512


def _is_rdna():
    """True if a ROCm device with RDNA WMMA (gfx11/gfx12) is present."""
    if not tvm_tirx.rocm(0).exist:
        return False
    from tvm_tirx.support import rocm  # pylint: disable=import-outside-toplevel

    if hasattr(rocm, "have_rdna_wmma"):
        return rocm.have_rdna_wmma()
    major = str(tvm_tirx.rocm(0).compute_version).split(".", maxsplit=1)[0]
    return major in ("11", "12")


def matmul(b_transposed):
    a = te.placeholder((M, K), name="A", dtype="float16")
    b = te.placeholder((N, K) if b_transposed else (K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(
        (M, N),
        lambda i, j: te.sum(
            tvm_tirx.tirx.Cast("float32", a[i, k])
            * tvm_tirx.tirx.Cast("float32", b[(j, k) if b_transposed else (k, j)]),
            axis=[k],
        ),
        name="C",
    )
    return a, b, c


@pytest.mark.gpu
@pytest.mark.skipif(not _is_rdna(), reason="needs an RDNA (gfx11/12) GPU with WMMA")
@pytest.mark.parametrize("b_transposed", [False, True])
def test_f16f16f32_m16n16k16_wave32(b_transposed, tmp_path):
    def tile(mapping):
        return lambda i, j: (i // 16, j // 16, *mapping(i % 16, j % 16))

    index_map_A = tile(shared_16x16_to_local_32x16_layout_A)
    # B fragment holds a column per lane: [K, N] -> layout_B; [N, K] (transposed) -> same as A
    index_map_B = tile(
        shared_16x16_to_local_32x16_layout_A
        if b_transposed
        else shared_16x16_to_local_32x16_layout_B
    )
    index_map_C = tile(shared_16x16_to_local_32x8_layout_C)
    sch = mfma_schedule(
        te.create_function(matmul(b_transposed)),
        16,
        "float16",
        b_transposed,
        [1, 8, 2, 2, 1],
        [1, 8, 2, 1, 2],
        [16, 2, 1],
        index_map_A,
        index_map_B,
        index_map_C,
        ROCM_WMMA_LOAD_16x16_A_SHARED_F16_INTRIN,
        ROCM_WMMA_LOAD_16x16_B_TRANS_SHARED_F16_INTRIN
        if b_transposed
        else ROCM_WMMA_LOAD_16x16_B_SHARED_F16_INTRIN,
        ROCM_WMMA_F16F16F32_INTRIN,
        ROCM_WMMA_FILL_16x16_F32_INTRIN,
        ROCM_WMMA_STORE_16x16_F32_INTRIN,
        warp_size=32,
    )
    dev = tvm_tirx.rocm(0)
    # mcpu is detected from the device (rocm target canonicalizer); RDNA runs wave32.
    lib = tvm.compile(sch.mod["main"], target={"kind": "rocm", "thread_warp_size": 32})
    lib.export_library(str(tmp_path / "wmma.so"))  # load as a shared library (no ORC JIT needed)
    f = tvm_tirx.runtime.load_module(str(tmp_path / "wmma.so"))["main"]
    rng = np.random.default_rng(0)
    a_np = rng.uniform(size=(M, K)).astype("float16")
    b_np = rng.uniform(size=(N, K) if b_transposed else (K, N)).astype("float16")
    ref = a_np.astype("float32") @ (
        b_np.astype("float32").T if b_transposed else b_np.astype("float32")
    )
    a = tvm_tirx.runtime.tensor(a_np, dev)
    b = tvm_tirx.runtime.tensor(b_np, dev)
    c = tvm_tirx.runtime.tensor(np.zeros((M, N), "float32"), dev)
    f(a, b, c)
    dev.sync()
    tvm_tirx.testing.assert_allclose(c.numpy(), ref, rtol=1e-2, atol=1e-2)
