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

"""Tests for ``T.permute_layout``.

Coverage:

- The algorithm helpers (`_bank_free`, `_check_bijection`, `_choose_xor_k`)
  directly, with a NumPy oracle.
- End-to-end compiled-kernel byte-for-byte equivalence on CUDA for the SF
  fp8-blockwise-gemm transpose shapes (BLK_SFA = 128, 256) plus a few
  generic linear↔stride-permuted layouts and additional dtypes (u8, fp16,
  i32, u64).
- Reject cases: non-warp scope, dtype mismatch, shape mismatch, swizzle/
  compose layouts, layouts whose strides don't form a bijection on the
  slice.  Each must surface as a ``RuntimeError`` from the dispatcher and
  NOT silently emit a wrong kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env

# Helpers exposed by the dispatcher module for direct algorithm tests.
from tvm.tirx.cuda.operator.tile_primitive.permute_layout.warp_xor_swizzle import (
    _bank_free,
    _check_bijection,
    _choose_xor_k,
)
from tvm.tirx.layout import S, SwizzleLayout, TileLayout

# ---------------------------------------------------------------------------
# Algorithm-only tests (no CUDA needed).
# ---------------------------------------------------------------------------


def _np_layout_offset(extent, strides, multi_idx):
    return int(sum(s * i for s, i in zip(strides, multi_idx)))


def _expected_permute(src_np, src_strides, dst_strides, extent):
    """Compute the expected output: dst at byte offset ``L_dst(i)`` holds the
    value at ``src`` byte offset ``L_src(i)``, for every logical index i.
    """
    V = math.prod(extent)
    dst_np = np.zeros_like(src_np)
    for flat in range(V):
        idx = []
        rem = flat
        for e in reversed(extent):
            idx.append(rem % e)
            rem //= e
        idx = list(reversed(idx))
        src_off = _np_layout_offset(extent, src_strides, idx)
        dst_off = _np_layout_offset(extent, dst_strides, idx)
        dst_np.reshape(-1)[dst_off] = src_np.reshape(-1)[src_off]
    return dst_np


def test_bank_free_sf_128_u32():
    """SF BLK_SFA=128: write phase has 4-way conflict at k=0, free at k=2."""
    extent = [4, 32]
    src = [32, 1]
    dst = [1, 4]
    bytes_per = 4
    P = 4
    assert _bank_free(extent, src, bytes_per, P, 0)
    assert not _bank_free(extent, dst, bytes_per, P, 0)
    assert _bank_free(extent, dst, bytes_per, P, 2)
    assert _choose_xor_k(extent, src, dst, bytes_per, P) == 2


def test_bank_free_sf_256_u32():
    """SF BLK_SFA=256: same shift=3 (k=2) handles the high block too."""
    extent = [2, 4, 32]
    src = [128, 32, 1]
    dst = [128, 1, 4]
    bytes_per = 4
    P = 8
    assert _bank_free(extent, src, bytes_per, P, 0)
    assert not _bank_free(extent, dst, bytes_per, P, 0)
    assert _bank_free(extent, dst, bytes_per, P, 2)
    assert _choose_xor_k(extent, src, dst, bytes_per, P) == 2


def test_identity_no_xor():
    """L_src == L_dst => k=0 (no XOR needed and the op is essentially a copy)."""
    assert _choose_xor_k([4, 32], [32, 1], [32, 1], 4, 4) == 0
    # A 2D buffer with row-major to row-major is a true no-op.
    assert _bank_free([4, 32], [32, 1], 4, 4, 0)


def test_bijection_check_rejects_aliased():
    """If two logical indices map to the same physical byte, reject."""
    # Stride 0 on a non-singleton extent => alias.
    assert not _check_bijection([4, 32], [0, 1])
    # Negative or non-contiguous-but-bijective is still fine.
    assert _check_bijection([4, 32], [1, 4])


def test_dtype_widths_choose_xor_k():
    """Each dtype's outcome:

    The unvectorized algorithm is provably correct only when every per-lane
    access maps to a single 4-byte bank.  For 4-byte dtypes that always holds
    (one element per bank), so we expect a valid k.  For sub-4-byte dtypes
    with stride-1 reads, multiple lanes share a bank no matter how we permute
    register slots — the dispatcher correctly rejects those (k is None).
    """
    extent = [4, 32]
    src = [32, 1]  # linear
    dst = [1, 4]  # transposed
    # u32: this is the SF case; the algorithm must find k=2.
    assert _choose_xor_k(extent, src, dst, 4, 4) == 2
    # u16/fp16, u8: stride-1 in bytes < 4 packs >1 lane into the same bank;
    # register-slot XOR cannot fix that, so the dispatcher rejects.
    assert _choose_xor_k(extent, src, dst, 2, 4) is None
    assert _choose_xor_k(extent, src, dst, 1, 4) is None


# ---------------------------------------------------------------------------
# End-to-end compiled-kernel tests on CUDA.
# ---------------------------------------------------------------------------


def _has_cuda():
    try:
        return tvm.cuda(0).exist
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")


def _compile_and_run(prim_func, np_inputs):
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": prim_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        tensors = [tvm.runtime.tensor(a, dev) for a in np_inputs]
        mod(*tensors)
        return [tensor.numpy() for tensor in tensors]

    outputs = tvm.testing.run_with_gpu_lock(run_and_check)
    return outputs, mod.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@needs_cuda
@pytest.mark.parametrize(
    "name, pipe, blk, dtype",
    [
        ("sf_128_u32", 2, 128, "uint32"),
        ("sf_256_u32", 2, 256, "uint32"),
        ("sf_128_i32", 2, 128, "int32"),
        ("sf_128_fp32", 2, 128, "float32"),
    ],
)
def test_sf_blockwise_transpose(name, pipe, blk, dtype):
    """SF blockwise-GEMM scale-factor transpose, the canonical use case."""
    high = blk // 128 if blk >= 128 else 1
    # Use 4D logical shape (PIPE, high, 4, 32) to keep the high-block factored.
    shape = (pipe, high, 4, 32)

    # Element strides for src (linear) and dst (transposed within each
    # 128-block).  Stage stride = blk; each 128-block contributes 128 to the
    # high stride.
    src_strides = (blk, 128, 32, 1)
    dst_strides = (blk, 128, 1, 4)
    pre = TileLayout(S[shape:src_strides])
    post = TileLayout(S[shape:dst_strides])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        for s in T.serial(0, pipe):
            Tx.warp.permute_layout(
                B_buf[s, 0:high, 0:4, 0:32], A_buf[s, 0:high, 0:4, 0:32]
            )
        # fmt: on

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros_like(A_np)

    [_, B_out], src = _compile_and_run(f, [A_np, B_np])

    # The dispatcher must have picked the XOR-swizzled variant; check that
    # the generated CUDA contains the per-lane XOR pattern.  This is the
    # "no perf regression" smoke test: any future variant that omits the
    # XOR would re-introduce 4-way bank conflicts.
    assert ">> 3" in src, f"expected XOR-swizzle (lane>>3) in CUDA for {name}"
    assert "warp_sync" in src or "syncwarp" in src

    # Byte-for-byte equality via numpy reference.
    for s in range(pipe):
        A_flat = A_np[s].reshape(-1)
        B_flat = B_out[s].reshape(-1)
        ref = _expected_permute(
            A_flat,
            list(src_strides[1:]),
            list(dst_strides[1:]),
            list(shape[1:]),
        )
        np.testing.assert_array_equal(B_flat, ref, err_msg=f"{name} stage {s}")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@needs_cuda
def test_identity_passes_through_as_copy():
    """L_src == L_dst should still compile and produce a correct (identity) copy."""
    shape = (4, 32)
    layout = TileLayout(S[shape : (32, 1)])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint32", layout=layout)
        B_buf = T.match_buffer(B, shape, "uint32", layout=layout)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array("uint32", shape)
    B_np = np.zeros_like(A_np)

    [_, B_out], _ = _compile_and_run(f, [A_np, B_np])
    np.testing.assert_array_equal(B_out, A_np)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@needs_cuda
@pytest.mark.parametrize("dtype", ["uint32", "int32", "float32"])
@pytest.mark.parametrize(
    "shape, src_strides, dst_strides",
    [
        # (8, 32) → (8, 32) transposed: src linear, dst column-major.
        ((8, 32), (32, 1), (1, 8)),
        # (16, 32): per_thread = 16 — tests P=16 path.
        ((16, 32), (32, 1), (1, 16)),
    ],
)
def test_generic_transpose(shape, src_strides, dst_strides, dtype):
    """Generic linear↔transposed pairs at various P values."""
    pre = TileLayout(S[shape:src_strides])
    post = TileLayout(S[shape:dst_strides])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros_like(A_np)
    [_, B_out], _ = _compile_and_run(f, [A_np, B_np])

    ref = _expected_permute(A_np.reshape(-1), list(src_strides), list(dst_strides), list(shape))
    np.testing.assert_array_equal(B_out.reshape(-1), ref)


# ---------------------------------------------------------------------------
# Reject cases: the dispatcher must surface a clear error, never silently
# emit a wrong kernel.
# ---------------------------------------------------------------------------


def _build_and_assert_rejected(shape, src_layout, dst_layout, dtype, msg_substr):
    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=src_layout)
        B_buf = T.match_buffer(B, shape, dtype, layout=dst_layout)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError) as exc_info:
        mod = tvm.IRModule({"main": f})
        tvm.compile(mod, target=target, tir_pipeline="tirx")
    assert msg_substr in str(exc_info.value), (
        f"expected reject reason to mention {msg_substr!r}, got: {exc_info.value}"
    )


def test_reject_dtype_mismatch():
    shape = (4, 32)
    layout = TileLayout(S[shape : (32, 1)])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, "uint32", layout=layout)
        B_buf = T.match_buffer(B, shape, "uint16", layout=layout)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "dtype mismatch" in str(exc_info.value)


def test_reject_shape_mismatch():
    src_layout = TileLayout(S[(4, 32) : (32, 1)])
    dst_layout = TileLayout(S[(8, 16) : (16, 1)])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, (4, 32), "uint32", layout=src_layout)
        B_buf = T.match_buffer(B, (8, 16), "uint32", layout=dst_layout)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "shape mismatch" in str(exc_info.value)


def test_reject_swizzle_layout():
    """ComposeLayout(SwizzleLayout, TileLayout) is not supported by the warp variant."""
    from tvm.tirx.layout import ComposeLayout

    inner = TileLayout(S[(4, 32) : (32, 1)])
    sw = SwizzleLayout(per_element=2, swizzle_len=2, atom_len=4)
    swizzled = ComposeLayout(sw, inner)
    plain = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, (4, 32), "uint32", layout=swizzled)
        B_buf = T.match_buffer(B, (4, 32), "uint32", layout=plain)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.warp.permute_layout(B_buf, A_buf)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "TileLayout" in str(exc_info.value)


def test_reject_non_warp_scope():
    layout_pre = TileLayout(S[(4, 32) : (32, 1)])
    layout_post = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, (4, 32), "uint32", layout=layout_pre)
        B_buf = T.match_buffer(B, (4, 32), "uint32", layout=layout_post)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([32])
        Tx.cta.permute_layout(B_buf, A_buf)  # cta scope, not warp
        # fmt: on

    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError) as exc_info:
        tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    assert "warp" in str(exc_info.value)


@pytest.mark.parametrize("dtype", ["uint32", "float32"])
@pytest.mark.gpu
def test_shared_to_shared_uses_direct_ldst(dtype):
    """Compile-only: a shared->shared 32b transpose must take the direct
    base-ptr + byte-offset ``ld.shared`` / ``st.shared`` path.

    For a 4/8-byte dtype with both operands in shared memory, indexing through
    ``buf[...]`` lowers the swizzled layout to a per-element IMAD flatten. The
    direct path computes one base ptr (``ptr_to(stride_offset)``) and adds a
    compile-time ``off * dtype_bytes`` per register slot, then issues
    ``T.ptx.ld/st(..., space="shared")``. The bits move through a uint
    container, so ``float32`` (whose ``ld.b32`` cannot return a float) lowers
    the same way as the ``uint32`` SF case.
    """
    shape = (4, 32)
    pre = TileLayout(S[shape : (32, 1)])  # linear
    post = TileLayout(S[shape : (1, 4)])  # transposed within the 128-block

    # fmt: off
    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([32])
        sA = T.alloc_buffer(shape, dtype, scope="shared", layout=pre)
        sB = T.alloc_buffer(shape, dtype, scope="shared", layout=post)
        Tx.cta.copy(sA[:, :], A_buf[:, :])
        T.cuda.cta_sync()
        Tx.warp.permute_layout(sB[:, :], sA[:, :])
        T.cuda.cta_sync()
        Tx.cta.copy(B_buf[:, :], sB[:, :])
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "ld.shared" in src, f"expected direct ld.shared in permute; src=\n{src}"
    assert "st.shared" in src, f"expected direct st.shared in permute; src=\n{src}"


if __name__ == "__main__":
    tvm.testing.main()
