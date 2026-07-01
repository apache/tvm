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
"""Tests for FMA op dispatch, layout=None local dispatch, scalar broadcast,
and rounding mode support."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout, wg_local_layout


def _get_sm_version():
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        return 0
    digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# FMA op: scalar scale + scalar bias
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_scalar_scalar():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 128
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    scale_val = 0.5
    bias_val = -1.0

    @T.prim_func
    def test_func(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        T.device_entry()
        _bx = T.cta_id([1])
        tx = T.thread_id([N])
        buf = T.alloc_buffer((1,), dtype, scope="local", layout=TileLayout(S[1]))
        Tx.copy(buf, A[tx : tx + 1])
        Tx.fma(buf, buf, T.float32(scale_val), T.float32(bias_val))
        Tx.copy(A[tx : tx + 1], buf)

    with target:
        A_np = np.random.rand(N).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A)
        expected = A_np * scale_val + bias_val
        tvm.testing.assert_allclose(expected, A.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# FMA op: buffer scale + scalar bias (Horner pattern)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_buffer_scale_scalar_bias():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    coeff = 0.695

    @T.prim_func
    def test_func(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        B = T.match_buffer(B_ptr, (N,), dtype, layout=TileLayout(S[N]))
        T.device_entry()
        _bx = T.cta_id([1])
        _tx = T.thread_id([1])
        acc = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        frac = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        Tx.copy(acc, A[0:N])
        Tx.copy(frac, B[0:N])
        Tx.fma(acc, acc, frac, T.float32(coeff))
        Tx.copy(A[0:N], acc)

    with target:
        A_np = np.random.rand(N).astype(dtype)
        B_np = np.random.rand(N).astype(dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A, B)
        expected = A_np * B_np + coeff
        tvm.testing.assert_allclose(expected, A.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# Binary op with scalar broadcast (Expr scalar, e.g. BufferLoad)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_mul_scalar_broadcast():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed mul requires sm_100+, got sm_{sm}")

    N = 16
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    @T.prim_func
    def test_func(A_ptr: T.handle, S_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        Scale = T.match_buffer(S_ptr, (1,), dtype, layout=TileLayout(S[1]))
        T.device_entry()
        _bx = T.cta_id([1])
        _tx = T.thread_id([1])
        a_local = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        s_local = T.alloc_buffer((1,), dtype, scope="local", layout=TileLayout(S[1]))
        Tx.copy(a_local, A[0:N])
        Tx.copy(s_local, Scale[0:1])
        Tx.mul(a_local, a_local, s_local[0])
        Tx.copy(A[0:N], a_local)

    with target:
        A_np = np.random.rand(N).astype(dtype)
        S_np = np.array([2.5], dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        S_dev = tvm.runtime.tensor(S_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A_dev, S_dev)
        expected = A_np * S_np[0]
        tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# Binary add with rounding mode
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_add_rounding_mode():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed add with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    round_const = float(2**23 + 2**22)

    @T.prim_func
    def test_func(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        T.device_entry()
        _bx = T.cta_id([1])
        _tx = T.thread_id([1])
        buf = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        Tx.copy(buf, A[0:N])
        Tx.add(buf, buf, T.float32(round_const), rounding_mode="rm")
        Tx.copy(A[0:N], buf)

    with target:
        A_np = np.array([1.3, 2.7], dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        # Check that the PTX uses the rounding mode
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"add\.rm\.ftz\.f32x2", src) or re.search(
            r"tvm_builtin_ptx_add_packed_", src
        ), f"Expected packed add with rm rounding in PTX:\n{src}"
        mod(A_dev)
        expected = A_np + round_const
        tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1.0)


# ---------------------------------------------------------------------------
# FMA op: layout=None local buffer (no TileLayout)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_no_layout():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 4
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    scale_val = 2.0
    bias_val = 1.0

    @T.prim_func
    def test_func(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        T.device_entry()
        _bx = T.cta_id([1])
        _tx = T.thread_id([1])
        buf = T.alloc_local([N], dtype)
        for i in T.serial(N):
            buf[i] = A[i]
        Tx.fma(buf[0:N], buf[0:N], T.float32(scale_val), T.float32(bias_val))
        for i in T.serial(N):
            A[i] = buf[i]

    with target:
        A_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A_dev)
        expected = A_np * scale_val + bias_val
        tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)


# ---------------------------------------------------------------------------
# Binary sub with rounding mode (buffer-buffer)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_sub_buffer_buffer_rounding():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed sub with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    @T.prim_func
    def test_func(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (N,), dtype, layout=TileLayout(S[N]))
        B = T.match_buffer(B_ptr, (N,), dtype, layout=TileLayout(S[N]))
        T.device_entry()
        _bx = T.cta_id([1])
        _tx = T.thread_id([1])
        a_buf = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        b_buf = T.alloc_buffer((N,), dtype, scope="local", layout=TileLayout(S[N]))
        Tx.copy(a_buf, A[0:N])
        Tx.copy(b_buf, B[0:N])
        Tx.sub(a_buf, a_buf, b_buf, rounding_mode="rn")
        Tx.copy(A[0:N], a_buf)

    with target:
        A_np = np.array([3.14, 2.71], dtype=dtype)
        B_np = np.array([1.41, 0.57], dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        B_dev = tvm.runtime.tensor(B_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"sub\.rn\.ftz\.f32x2", src) or re.search(
            r"tvm_builtin_ptx_sub_packed_", src
        ), f"Expected packed sub with rn rounding in PTX:\n{src}"
        mod(A_dev, B_dev)
        expected = A_np - B_np
        tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_warpgroup_wg_local_layout():
    rows, cols = 128, 8
    dtype = "float32"
    scale_val = 1.5
    bias_val = -0.25
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    @T.prim_func
    def test_func(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        B = T.match_buffer(B_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        T.device_entry()
        _bx = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid = T.thread_id_in_wg([rows])

        reg = T.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
        reg_row = reg.local(cols)
        for i in T.serial(cols):
            reg_row[i] = A[tid, i]
        Tx.wg.fma(reg, reg, T.float32(scale_val), T.float32(bias_val))
        reg_row_1 = reg.local(cols)
        for i in T.serial(cols):
            B[tid, i] = reg_row_1[i]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.zeros((rows, cols), dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        B_dev = tvm.runtime.tensor(B_np, dev)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A_dev, B_dev)
        expected = A_np * scale_val + bias_val
        tvm.testing.assert_allclose(expected, B_dev.numpy(), atol=1e-5)


# -----------------------------------------------------------------------------
# Dispatch codegen check (no GPU runtime — explicit target arch).
# Complements ``test_fma_warpgroup_wg_local_emits_packed_f32x2`` (which uses
# the host-detected ``Target("cuda")`` and skips when arch < sm_100).
# -----------------------------------------------------------------------------
def test_fma_f32_sm100_packed_f32x2_dispatch():
    """fma f32 + all-local → reg.py + fma_f32x2 packed (no T.vectorized)."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float32", layout=lay)
        B = T.match_buffer(B_ptr, shape, "float32", layout=lay)
        C = T.match_buffer(C_ptr, shape, "float32", layout=lay)
        D = T.match_buffer(D_ptr, shape, "float32", layout=lay)
        T.device_entry()
        _bx = T.cta_id([1])
        tx = T.thread_id([64])
        ra = T.alloc_buffer(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))
        rb = T.alloc_buffer(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))
        rc = T.alloc_buffer(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))
        rd = T.alloc_buffer(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))
        Tx.copy(ra, A[tx])
        Tx.copy(rb, B[tx])
        Tx.copy(rc, C[tx])
        Tx.fma(rd, ra, rb, rc)
        Tx.copy(D[tx], rd)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert re.search(r"fma\.[a-z]+\.ftz\.f32x2", src) or re.search(
        r"tvm_builtin_ptx_fma_packed_", src
    ), f"expected packed fma_f32x2; got:\n{src[:2000]}"


if __name__ == "__main__":
    tvm.testing.main()
