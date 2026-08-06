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
"""Codegen tests for Ampere (sm_80) warp-level ``mma.sync`` tensor cores.

These exercise the ``T.ptxd.mma`` chain directly (not via the gemm
dispatch). ``ptx.mma`` takes one pointer per 32-bit register for each operand
(``d_ptrs`` / ``a_ptrs`` / ``b_ptrs`` / ``c_ptrs``), enumerated in the fixed
PTX register order, so the b32 registers may be scattered in the register file
while the two packed fp16/bf16 within a b32 stay contiguous. For m16n8k{8,16}
with f32 accumulation the per-lane register counts are:

    A: 2 inputs per b32 -> k16: 4 b32 (regs 0,2,4,6); k8: 2 b32 (regs 0,2)
    B: 2 inputs per b32 -> k16: 2 b32 (regs 0,2);     k8: 1 b32 (reg 0)
    D/C: 4 f32 accumulator registers (0,1,2,3)
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


def _get_source(func: tvm.tirx.PrimFunc):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def _np_in(dtype):
    if dtype == "bfloat16":
        return __import__("ml_dtypes").bfloat16
    return np.float16


def _run_mma(mod, K, no_c_ptr, np_in):
    """Run an m16n8kK mma kernel and check D == A @ B (+ C) against numpy."""
    np.random.seed(0)
    A_np = np.random.randn(16, K).astype(np_in)
    B_np = np.random.randn(K, 8).astype(np_in)
    C_np = np.random.randn(16, 8).astype(np.float32)
    ref = A_np.astype(np.float32) @ B_np.astype(np.float32)
    if not no_c_ptr:
        ref = ref + C_np

    def run_and_check():
        dev = tvm.cuda()
        D = tvm.runtime.tensor(np.zeros((16, 8), np.float32), device=dev)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        C = tvm.runtime.tensor(C_np, device=dev)
        mod(D, A, B, C)
        np.testing.assert_allclose(D.numpy(), ref, atol=1e-2, rtol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("a_type", ["float16", "bfloat16"])
@pytest.mark.parametrize("no_c_ptr", [False, True])
def test_ptx_mma_m16n8k16(a_type, no_c_ptr):
    """m16n8k16 row.col mma, f32 accumulate: A is 16x16 (4 b32/lane), B is 16x8
    as [K, N] (2 b32/lane), D/C is 16x8 (4 f32/lane)."""
    if a_type == "bfloat16":
        pytest.importorskip("ml_dtypes")
    b_type = a_type

    _elem = "f16" if a_type == "float16" else "bf16"

    # fmt: off
    @T.prim_func
    def main(
        D: T.Buffer((16, 8), "float32"),
        A: T.Buffer((16, 16), a_type),
        B: T.Buffer((16, 8), b_type),
        C: T.Buffer((16, 8), "float32"),
    ):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        D_local = T.alloc_local([4], "float32")
        A_local = T.alloc_local([8], a_type)
        B_local = T.alloc_local([4], b_type)
        C_local = T.alloc_local([4], "float32")

        @T.inline
        def G2L(buf_local, buf_global, block_8x8, mode="row"):
            if mode == "row":
                for i in range(block_8x8):
                    row = T.meta_var(i % 2 * 8 + tx // 4)
                    col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                    for j in range(2):
                        buf_local[i * 2 + j] = buf_global[row, col + j]
            elif mode == "col":
                for i in range(block_8x8):
                    row = T.meta_var(i % 2 * 8 + (tx % 4) * 2)
                    col = T.meta_var(i // 2 * 8 + tx // 4)
                    for j in range(2):
                        buf_local[i * 2 + j] = buf_global[row + j, col]

        G2L(D_local, D, 2)
        G2L(A_local, A, 4)
        G2L(B_local, B, 2, "col")
        G2L(C_local, C, 2)

        # One register per b32, in PTX order: A=4, B=2, D/C=4. The 16-bit
        # multiplicands are packed two per b32, so they ride a uint32 view.
        A_words = A_local.view("uint32")
        B_words = B_local.view("uint32")
        # ptxd takes the accumulator as an operand; the legacy "omit c" form
        # fed literal zeros, which is now spelled at the call site.
        if no_c_ptr:
            for i in range(4):
                C_local[i] = T.float32(0)
        T.ptxd[f"mma.sync.aligned.m16n8k16.row.col.f32.{_elem}.{_elem}.f32"](
            *[D_local[i] for i in range(4)],
            *[A_words[i] for i in range(4)],
            *[B_words[i] for i in range(2)],
            *[C_local[i] for i in range(4)],
        )

        for i in range(2):
            row = T.meta_var(i % 2 * 8 + tx // 4)
            col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
            for j in range(2):
                D[row, col + j] = D_local[i * 2 + j]
    # fmt: on

    src, mod = _get_source(main)
    assert "mma.sync.aligned.m16n8k16.row.col" in src
    _run_mma(mod, 16, no_c_ptr, _np_in(a_type))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("a_type", ["float16", "bfloat16"])
@pytest.mark.parametrize("no_c_ptr", [False, True])
def test_ptx_mma_m16n8k8(a_type, no_c_ptr):
    """m16n8k8 row.col mma, f32 accumulate: A is 16x8 (2 b32/lane), B is 8x8
    as [K, N] (1 b32/lane), D/C is 16x8 (4 f32/lane)."""
    if a_type == "bfloat16":
        pytest.importorskip("ml_dtypes")
    b_type = a_type

    _elem = "f16" if a_type == "float16" else "bf16"

    # fmt: off
    @T.prim_func
    def main(
        D: T.Buffer((16, 8), "float32"),
        A: T.Buffer((16, 8), a_type),
        B: T.Buffer((8, 8), b_type),
        C: T.Buffer((16, 8), "float32"),
    ):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        D_local = T.alloc_local([4], "float32")
        A_local = T.alloc_local([4], a_type)
        B_local = T.alloc_local([2], b_type)
        C_local = T.alloc_local([4], "float32")

        @T.inline
        def G2L(buf_local, buf_global, block_8x8, mode="row"):
            if mode == "row":
                for i in range(block_8x8):
                    row = T.meta_var(i % 2 * 8 + tx // 4)
                    col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
                    for j in range(2):
                        buf_local[i * 2 + j] = buf_global[row, col + j]
            elif mode == "col":
                for i in range(block_8x8):
                    row = T.meta_var(i % 2 * 8 + (tx % 4) * 2)
                    col = T.meta_var(i // 2 * 8 + tx // 4)
                    for j in range(2):
                        buf_local[i * 2 + j] = buf_global[row + j, col]

        G2L(D_local, D, 2)
        G2L(A_local, A, 2)
        G2L(B_local, B, 1, "col")
        G2L(C_local, C, 2)

        # One register per b32, in PTX order: A=2, B=1, D/C=4.
        A_words = A_local.view("uint32")
        B_words = B_local.view("uint32")
        # ptxd takes the accumulator as an operand; the legacy "omit c" form
        # fed literal zeros, which is now spelled at the call site.
        if no_c_ptr:
            for i in range(4):
                C_local[i] = T.float32(0)
        T.ptxd[f"mma.sync.aligned.m16n8k8.row.col.f32.{_elem}.{_elem}.f32"](
            *[D_local[i] for i in range(4)],
            *[A_words[i] for i in range(2)],
            *[B_words[i] for i in range(1)],
            *[C_local[i] for i in range(4)],
        )

        for i in range(2):
            row = T.meta_var(i % 2 * 8 + tx // 4)
            col = T.meta_var(i // 2 * 8 + (tx % 4) * 2)
            for j in range(2):
                D[row, col + j] = D_local[i * 2 + j]
    # fmt: on

    src, mod = _get_source(main)
    assert "mma.sync.aligned.m16n8k8.row.col" in src
    _run_mma(mod, 8, no_c_ptr, _np_in(a_type))


if __name__ == "__main__":
    tvm.testing.main()
