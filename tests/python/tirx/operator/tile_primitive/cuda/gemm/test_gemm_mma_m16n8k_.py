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
"""Tests for the CUDA synchronous ``gemm`` (mma.sync) tensor-core dispatch.

The dispatch lowers ``tirx.tile.gemm`` over pure-register fragments to warp-level
``mma.sync.aligned.m16n8k16/k8`` for bf16/f16 inputs with f32 accumulation.

The fragment layouts below are the standard m16n8 register maps (PTX ISA
§9.7.13; see ``tests/python/tirx-base/test_tir_ptx_mma.py``):

    lane = 4*g + t        (g = lane >> 2 in [0, 8),  t = lane & 3 in [0, 4))
    D/C[M, N]: M = g + 8*rM,  N = 2*t + rN,          c_id = 2*rM + rN
    A[M, K]:   M = g + 8*rM,  K = 2*t + p + 8*kHi,   ma   = p + 2*rM + 4*kHi
    B[K, N]:   N = g,         K = 2*t + p + 8*kHi,   mb   = p + 2*kHi

Most assertions run the CPU-only ``LowerTIRx`` transform; the numerical check
is guarded by ``requires_cuda`` since it needs a real device.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout, laneid
from tvm.tirx.operator.tile_primitive import list_registered_schedules

# Single-tile m16n8k8 fragment layouts -- the smallest unit everything else is
# built from. A is 16x8, B is 8x8 as [K, N], D/C is 16x8 (the accumulator does
# not depend on K). Every other layout (the k16 single tile, all tilings, and
# the transposed orientations) is derived from these three via ``tile_to`` /
# ``group`` + ``permute_by_groups``.
D_FRAG = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
A_FRAG_K8 = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
B_FRAG_K8 = TileLayout(S[(4, 2, 8) : (1 @ laneid, 1, 4 @ laneid)])
# m16n8k16 single tile = two k8 tiles stacked along K.
A_FRAG = A_FRAG_K8.tile_to([16, 16], [16, 8])
B_FRAG = B_FRAG_K8.tile_to([16, 8], [8, 8])


def _transpose_frag(layout, shape):
    """Swap the two logical axes of a 2D fragment layout.

    The transposed input orientations (A as [K, M], B as [N, K]) hold the exact
    same per-lane/per-register element distribution as the K-major fragments --
    only the buffer's logical axes are swapped. So instead of writing them out
    by hand, derive them: ``group`` the shard into the logical dims, then
    ``permute_by_groups`` to exchange the two groups.
    """
    grouped, seps = layout.group(shape)
    return grouped.permute_by_groups(seps, [1, 0])


# Transposed input orientations of the same single tile: A as [K, M], B as
# [N, K]. The dispatch swaps axes per the transpose flags; the .row.col mma is
# unchanged.
A_KM_FRAG = _transpose_frag(A_FRAG, [16, 16])
B_NK_FRAG = _transpose_frag(B_FRAG, [16, 8])


def _frag(Mt, Nt, Kt, kinst):
    """Fragment layouts for an Mt x Nt x Kt tiling of m16n8k{8,16}.

    Logical shapes: A = (16*Mt, kinst*Kt), B = (kinst*Kt, 8*Nt), D/C = (16*Mt, 8*Nt).
    Each operand's tiled layout is the single-tile base ``tile_to`` the full
    logical shape -- ``tile_to`` repeats the base's per-lane/per-register element
    map over the tile grid, so a tiling is just a grid of the single-tile
    fragments (the base is the single source of truth, k8 and k16 alike).
    """
    A_base = A_FRAG if kinst == 16 else A_FRAG_K8
    B_base = B_FRAG if kinst == 16 else B_FRAG_K8
    D = D_FRAG.tile_to([16 * Mt, 8 * Nt], [16, 8])
    A = A_base.tile_to([16 * Mt, kinst * Kt], [16, kinst])
    B = B_base.tile_to([kinst * Kt, 8 * Nt], [kinst, 8])
    return D, A, B


def _build_tiled(Mt, Nt, Kt, kinst, *, beta=0.0, dtype="float16", store=False):
    """A single-warp kernel issuing one ``T.gemm`` over an Mt x Nt x Kt tiling.

    With ``store=True`` the result is written back to a global buffer (a full
    kernel for codegen); otherwise only the ``T.gemm`` is emitted (for
    ``LowerTIRx`` dispatch checks).
    """
    Dl, Al, Bl = _frag(Mt, Nt, Kt, kinst)
    M, N, K = 16 * Mt, 8 * Nt, kinst * Kt

    if not store:

        @T.prim_func
        def gemm():
            T.device_entry()
            _cta = T.cta_id([1])
            _warp = T.warp_id([1])
            _lane = T.lane_id([32])
            A = T.alloc_buffer((M, K), dtype, scope="local", layout=Al)
            B = T.alloc_buffer((K, N), dtype, scope="local", layout=Bl)
            C = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
            D = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
            Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=beta)

        return gemm

    @T.prim_func
    def gemm(D_ptr: T.handle):
        D_g = T.match_buffer(D_ptr, (M, N), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        A = T.alloc_buffer((M, K), dtype, scope="local", layout=Al)
        B = T.alloc_buffer((K, N), dtype, scope="local", layout=Bl)
        C = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        D = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=beta)
        # Decode D's per-thread registers (c = ((mt*Nt + nt)*2 + rM)*2 + rN)
        # back to logical (M, N) and store, exercising the whole tiling.
        D_reg = D.local(Mt * Nt * 4)
        for c in T.unroll(Mt * Nt * 4):
            rN = c % 2
            rM = (c // 2) % 2
            nt = (c // 4) % Nt
            mt = c // (4 * Nt)
            D_g[mt * 16 + lane // 4 + rM * 8, nt * 8 + (lane % 4) * 2 + rN] = D_reg[c]

    return gemm


def _build_gemm(alpha=1.0, beta=0.0, dtype="bfloat16"):
    """A single-warp kernel issuing one ``T.gemm`` over register fragments."""

    @T.prim_func
    def gemm_min():
        T.device_entry()
        _cta = T.cta_id([1])
        _tid = T.thread_id([32])
        D = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        C = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        A = T.alloc_buffer((16, 16), dtype, scope="local", layout=A_FRAG)
        B = T.alloc_buffer((16, 8), dtype, scope="local", layout=B_FRAG)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=alpha, beta=beta)

    return gemm_min


def _build_transpose(transpose_A, transpose_B, *, store=False):
    """Single m16n8k16 tile with the requested A/B input orientations."""
    Al = A_KM_FRAG if transpose_A else A_FRAG
    Bl = B_NK_FRAG if transpose_B else B_FRAG
    A_shape = (16, 16)  # [K, M] or [M, K] -- both 16x16 for one tile
    B_shape = (8, 16) if transpose_B else (16, 8)

    if not store:

        @T.prim_func
        def gemm():
            T.device_entry()
            _cta = T.cta_id([1])
            _warp = T.warp_id([1])
            _lane = T.lane_id([32])
            A = T.alloc_buffer(A_shape, "float16", scope="local", layout=Al)
            B = T.alloc_buffer(B_shape, "float16", scope="local", layout=Bl)
            C = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
            D = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
            Tx.warp.gemm(
                D,
                A,
                B,
                C,
                transpose_A=transpose_A,
                transpose_B=transpose_B,
                alpha=1.0,
                beta=0.0,
            )

        return gemm

    @T.prim_func
    def gemm(D_ptr: T.handle):
        D_g = T.match_buffer(D_ptr, (16, 8), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        A = T.alloc_buffer(A_shape, "float16", scope="local", layout=Al)
        B = T.alloc_buffer(B_shape, "float16", scope="local", layout=Bl)
        C = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        D = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        Tx.warp.gemm(
            D,
            A,
            B,
            C,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            alpha=1.0,
            beta=0.0,
        )
        D_reg = D.local(4)
        for c in T.unroll(4):
            D_g[lane // 4 + (c // 2) * 8, (lane % 4) * 2 + c % 2] = D_reg[c]

    return gemm


def _build_dtypes(a_dtype, b_dtype, c_dtype, d_dtype):
    """Single tile with explicit per-operand dtypes (for decline checks)."""

    @T.prim_func
    def gemm_min():
        T.device_entry()
        _cta = T.cta_id([1])
        _tid = T.thread_id([32])
        D = T.alloc_buffer((16, 8), d_dtype, scope="local", layout=D_FRAG)
        C = T.alloc_buffer((16, 8), c_dtype, scope="local", layout=D_FRAG)
        A = T.alloc_buffer((16, 16), a_dtype, scope="local", layout=A_FRAG)
        B = T.alloc_buffer((16, 8), b_dtype, scope="local", layout=B_FRAG)
        Tx.warp.gemm(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)

    return gemm_min


def _build_tiled_numeric(Mt, Nt, Kt, kinst, beta, dtype):
    """End-to-end ``T.gemm`` over an Mt x Nt x Kt tiling, with the A/B inputs
    loaded and the D output stored register-by-register.

    Fragments are indexed through their per-register multi-dim ``.local()`` views
    (the shard's non-lane dims, in shard order): A = [Mt, rM(2), Kt, kHi, kp],
    B = [Kt, kHi, kp, Nt], D/C = [Mt, rM(2), Nt, rN(2)]. The lane owns g = lane>>2
    and t = lane&3; within a tile M = mt*16 + rM*8 + g, N = nt*8 + t*2 + rN,
    K = kt*kinst + kHi*8 + t*2 + kp.
    """
    Dl, Al, Bl = _frag(Mt, Nt, Kt, kinst)
    M, N, K = 16 * Mt, 8 * Nt, kinst * Kt
    KP = 2
    kHi_n = kinst // (4 * KP)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (M, K), dtype)
        B_g = T.match_buffer(B_ptr, (K, N), dtype)
        C_g = T.match_buffer(C_ptr, (M, N), "float32")
        D_g = T.match_buffer(D_ptr, (M, N), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        A_f = T.alloc_buffer((M, K), dtype, scope="local", layout=Al)
        B_f = T.alloc_buffer((K, N), dtype, scope="local", layout=Bl)
        C_f = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        D_f = T.alloc_buffer((M, N), "float32", scope="local", layout=Dl)
        A_reg = A_f.local(Mt, 2, Kt, kHi_n, KP)
        for mt, rM, kt, kHi, kp in T.grid(Mt, 2, Kt, kHi_n, KP):
            A_reg[mt, rM, kt, kHi, kp] = A_g[
                mt * 16 + lane // 4 + 8 * rM,
                kt * kinst + kHi * 8 + 2 * (lane % 4) + kp,
            ]
        B_reg = B_f.local(Kt, kHi_n, KP, Nt)
        for kt, kHi, kp, nt in T.grid(Kt, kHi_n, KP, Nt):
            B_reg[kt, kHi, kp, nt] = B_g[
                kt * kinst + kHi * 8 + 2 * (lane % 4) + kp,
                nt * 8 + lane // 4,
            ]
        if beta == 1.0:
            C_reg = C_f.local(Mt, 2, Nt, 2)
            for mt, rM, nt, rN in T.grid(Mt, 2, Nt, 2):
                C_reg[mt, rM, nt, rN] = C_g[
                    mt * 16 + lane // 4 + 8 * rM, nt * 8 + 2 * (lane % 4) + rN
                ]
        Tx.warp.gemm(D_f, A_f, B_f, C_f, transpose_A=False, transpose_B=False, alpha=1.0, beta=beta)
        D_reg = D_f.local(Mt, 2, Nt, 2)
        for mt, rM, nt, rN in T.grid(Mt, 2, Nt, 2):
            D_g[mt * 16 + lane // 4 + 8 * rM, nt * 8 + 2 * (lane % 4) + rN] = D_reg[mt, rM, nt, rN]

    return gemm, M, N, K


def _build_transpose_numeric(transpose_A, transpose_B, dtype="float16"):
    """End-to-end single-tile ``T.gemm`` for one A/B input orientation.

    The transposed A fragment (``A_KM_FRAG``) carries its registers in the
    [kHi, kp, rM] shard order (vs [rM, kHi, kp] for the K-major ``A_FRAG``); B's
    register order ([kHi, kp]) is the same for both orientations. The buffer
    index axes swap with the orientation, but each register still holds the same
    logical (M, K) / (K, N) element.
    """
    Al = A_KM_FRAG if transpose_A else A_FRAG
    Bl = B_NK_FRAG if transpose_B else B_FRAG
    A_shape = (16, 16)
    B_shape = (8, 16) if transpose_B else (16, 8)

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, A_shape, dtype)
        B_g = T.match_buffer(B_ptr, B_shape, dtype)
        D_g = T.match_buffer(D_ptr, (16, 8), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        A_f = T.alloc_buffer(A_shape, dtype, scope="local", layout=Al)
        B_f = T.alloc_buffer(B_shape, dtype, scope="local", layout=Bl)
        D_f = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        A_reg = A_f.local(2, 2, 2)
        if transpose_A:
            # A_KM_FRAG register order is [kHi, kp, rM]; buffer is [K, M].
            for kHi, kp, rM in T.grid(2, 2, 2):
                A_reg[kHi, kp, rM] = A_g[2 * (lane % 4) + kp + 8 * kHi, lane // 4 + 8 * rM]
        else:
            # A_FRAG register order is [rM, kHi, kp]; buffer is [M, K].
            for rM, kHi, kp in T.grid(2, 2, 2):
                A_reg[rM, kHi, kp] = A_g[lane // 4 + 8 * rM, 2 * (lane % 4) + kp + 8 * kHi]
        B_reg = B_f.local(2, 2)
        if transpose_B:
            # B_NK_FRAG buffer is [N, K].
            for kHi, kp in T.grid(2, 2):
                B_reg[kHi, kp] = B_g[lane // 4, 2 * (lane % 4) + kp + 8 * kHi]
        else:
            for kHi, kp in T.grid(2, 2):
                B_reg[kHi, kp] = B_g[2 * (lane % 4) + kp + 8 * kHi, lane // 4]
        Tx.warp.gemm(
            D_f,
            A_f,
            B_f,
            D_f,
            transpose_A=transpose_A,
            transpose_B=transpose_B,
            alpha=1.0,
            beta=0.0,
        )
        D_reg = D_f.local(2, 2)
        for rM, rN in T.grid(2, 2):
            D_g[lane // 4 + 8 * rM, 2 * (lane % 4) + rN] = D_reg[rM, rN]

    return gemm


def _lower(func):
    with tvm.target.Target("cuda"):
        return tvm.tirx.transform.LowerTIRx()(tvm.IRModule({"main": func}))


def test_cuda_gemm_mma_variant_is_registered():
    # Importing tvm.tirx registers all per-target schedule variants. The new
    # synchronous CUDA mma path must show up for ("gemm", "cuda"). The registry
    # keys ops by their full name (``op.name`` == "tirx.tile.gemm").
    schedules = list_registered_schedules()
    cuda_gemm = schedules.get("tirx.tile.gemm", {}).get("cuda", [])
    assert "mma.m16n8k*" in cuda_gemm, (
        f"mma.m16n8k* not registered; tirx.tile.gemm schedules = {schedules.get('tirx.tile.gemm')}"
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cuda_gemm_mma_lowers_to_mma_sync(dtype):
    """beta=0: the dispatch clears D, then issues a single accumulating mma with
    the registers laid out in the fixed PTX fragment order."""
    script = _lower(_build_gemm(alpha=1.0, beta=0.0, dtype=dtype))["main"].script()

    assert "T.ptx.mma(" in script
    assert "m16n8k16" in script
    # beta == 0 clears the accumulator before the K loop.
    assert "T.float32(0" in script
    # D accumulator: c_id = 2*rM + rN -> regs 0..3.
    for r in range(4):
        assert f"d_local[{r}]" in script
    # A multiplicand: b32 = rM + 2*kHi (kHi outer) -> ma in {0, 2, 4, 6}.
    for r in (0, 2, 4, 6):
        assert f"a_local[{r}]" in script
    # B multiplicand: b32 = kHi -> mb in {0, 2}.
    for r in (0, 2):
        assert f"b_local[{r}]" in script


def test_cuda_gemm_mma_accumulates_c_when_beta_one():
    """beta=1: the accumulator is initialized by copying C instead of zeroing."""
    script = _lower(_build_gemm(alpha=1.0, beta=1.0))["main"].script()

    assert "T.ptx.mma(" in script
    assert "m16n8k16" in script
    # The init reads C into D; nothing is zeroed.
    assert "c_local[" in script
    assert "T.float32(0" not in script


def test_cuda_gemm_mma_rejects_nonunit_alpha():
    """alpha != 1 is unsupported (ptx mma has no scale); dispatch must fail."""
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=2.0, beta=0.0))


def test_cuda_gemm_mma_rejects_fractional_beta():
    """beta must be 0 or 1 (mma only accumulates 1*C); other values must fail."""
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_gemm(alpha=1.0, beta=0.5))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_gemm_mma_numerical(dtype):
    """End-to-end D = A @ B on a single m16n8k16 tile (one warp).

    A is [M, K] = [16, 16], B is [K, N] = [16, 8], D is [M, N] = [16, 8].

    The lane-distributed register fragments cannot be filled with a whole-tile
    ``T.copy`` (the per-thread axis can't be matched coordinate-wise), so each
    of a lane's registers is loaded/stored by decoding the m16n8k16 register map
    with ``g = lane >> 2`` and ``t = lane & 3``. The per-register *slot* order
    matches the dispatch's fragment register layout:

        A reg slot = 4*rM + 2*kHi + kp  -> M = g + 8*rM, K = 2*t + kp + 8*kHi
        B reg slot = 2*kHi + kp         -> K = 2*t + kp + 8*kHi, N = g
        D reg slot = 2*rM + rN          -> M = g + 8*rM, N = 2*t + rN
    """
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = ml_dtypes.bfloat16
    else:
        np_dtype = np.float16

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (16, 16), dtype)
        B_g = T.match_buffer(B_ptr, (16, 8), dtype)
        D_g = T.match_buffer(D_ptr, (16, 8), "float32")
        T.device_entry()
        _cta = T.cta_id([1])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        A_f = T.alloc_buffer((16, 16), dtype, scope="local", layout=A_FRAG)
        B_f = T.alloc_buffer((16, 8), dtype, scope="local", layout=B_FRAG)
        D_f = T.alloc_buffer((16, 8), "float32", scope="local", layout=D_FRAG)
        A_reg = A_f.local(8)
        for s in T.unroll(8):
            kp = s % 2
            kHi = (s // 2) % 2
            rM = s // 4
            A_reg[s] = A_g[lane // 4 + 8 * rM, 2 * (lane % 4) + kp + 8 * kHi]
        B_reg = B_f.local(4)
        for s in T.unroll(4):
            kp = s % 2
            kHi = s // 2
            B_reg[s] = B_g[2 * (lane % 4) + kp + 8 * kHi, lane // 4]
        Tx.warp.gemm(D_f, A_f, B_f, D_f, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)
        D_reg = D_f.local(4)
        for s in T.unroll(4):
            rN = s % 2
            rM = s // 2
            D_g[lane // 4 + 8 * rM, 2 * (lane % 4) + rN] = D_reg[s]

    dev = tvm.cuda(0)
    with tvm.target.Target("cuda"):
        mod = tvm.compile(tvm.IRModule({"main": gemm}), target="cuda", tir_pipeline="tirx")

    np.random.seed(0)
    A_np = np.random.uniform(-1, 1, (16, 16)).astype(np.float32)
    B_np = np.random.uniform(-1, 1, (16, 8)).astype(np.float32)
    A_dev = tvm.runtime.tensor(A_np.astype(np_dtype), dev)
    B_dev = tvm.runtime.tensor(B_np.astype(np_dtype), dev)
    D_dev = tvm.runtime.tensor(np.zeros((16, 8), np.float32), dev)
    mod(A_dev, B_dev, D_dev)

    golden = A_np @ B_np
    tvm.testing.assert_allclose(golden, D_dev.numpy(), atol=1e-2, rtol=1e-2)


# (Mt, Nt, Kt, kinst) tilings: single tile, each dim multi-tiled, fully tiled,
# M = 64, and the m16n8k8 (kHi == 1) variants including a non-16-divisible K.
_TILED_SHAPES = [
    (1, 1, 1, 16),  # single m16n8k16 tile
    (2, 1, 1, 16),  # two M-tiles
    (1, 2, 1, 16),  # two N-tiles
    (1, 1, 2, 16),  # two K-tiles (accumulated in place)
    (2, 2, 2, 16),  # every dim tiled
    (4, 1, 1, 16),  # M = 64
    (1, 1, 1, 8),  # m16n8k8 single tile (kHi == 1)
    (1, 1, 3, 8),  # K = 24 -> three k8 tiles
    (2, 2, 3, 8),  # k8, every dim tiled
]
# (dtype, beta) input modes crossed against every shape: f16/bf16 inputs, with
# beta = 0 (D = A @ B) and beta = 1 (D = A @ B + C, accumulating C in place).
_TILED_MODES = [
    ("float16", 0.0),
    ("bfloat16", 0.0),
    ("float16", 1.0),
    ("bfloat16", 1.0),
]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("Mt, Nt, Kt, kinst", _TILED_SHAPES)
@pytest.mark.parametrize("dtype, beta", _TILED_MODES)
def test_cuda_gemm_mma_numerical_tiled(dtype, beta, Mt, Nt, Kt, kinst):
    """End-to-end D = A @ B (+ C when beta==1) over an Mt x Nt x Kt tiling.

    The two stacked ``parametrize`` decorators form the cartesian product of
    every tiling shape with every (dtype, beta) input mode, so each combination
    is an independent pytest item (pytest-xdist runs them in parallel)."""
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = ml_dtypes.bfloat16
    else:
        np_dtype = np.float16

    func, M, N, K = _build_tiled_numeric(Mt, Nt, Kt, kinst, beta, dtype)
    dev = tvm.cuda(0)
    with tvm.target.Target("cuda"):
        mod = tvm.compile(tvm.IRModule({"main": func}), target="cuda", tir_pipeline="tirx")

    np.random.seed(0)
    A_np = np.random.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = np.random.uniform(-1, 1, (K, N)).astype(np.float32)
    C_np = np.random.uniform(-1, 1, (M, N)).astype(np.float32)
    A_dev = tvm.runtime.tensor(A_np.astype(np_dtype), dev)
    B_dev = tvm.runtime.tensor(B_np.astype(np_dtype), dev)
    C_dev = tvm.runtime.tensor(C_np, dev)
    D_dev = tvm.runtime.tensor(np.zeros((M, N), np.float32), dev)
    mod(A_dev, B_dev, C_dev, D_dev)

    golden = A_np @ B_np + (C_np if beta == 1.0 else 0.0)
    tvm.testing.assert_allclose(golden, D_dev.numpy(), atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cuda_gemm_mma_numerical_transpose(transpose_A, transpose_B, dtype):
    """End-to-end D = A @ B for every A/B input orientation, crossed with dtype.

    The orientation and dtype decorators form a cartesian product, so each
    (transpose_A, transpose_B, dtype) is an independent pytest item."""
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = ml_dtypes.bfloat16
    else:
        np_dtype = np.float16

    func = _build_transpose_numeric(transpose_A, transpose_B, dtype)
    dev = tvm.cuda(0)
    with tvm.target.Target("cuda"):
        mod = tvm.compile(tvm.IRModule({"main": func}), target="cuda", tir_pipeline="tirx")

    np.random.seed(0)
    A_log = np.random.uniform(-1, 1, (16, 16)).astype(np.float32)  # logical A[M, K]
    B_log = np.random.uniform(-1, 1, (16, 8)).astype(np.float32)  # logical B[K, N]
    A_buf = (A_log.T if transpose_A else A_log).astype(np_dtype)
    B_buf = (B_log.T if transpose_B else B_log).astype(np_dtype)
    A_dev = tvm.runtime.tensor(A_buf, dev)
    B_dev = tvm.runtime.tensor(B_buf, dev)
    D_dev = tvm.runtime.tensor(np.zeros((16, 8), np.float32), dev)
    mod(A_dev, B_dev, D_dev)

    tvm.testing.assert_allclose(A_log @ B_log, D_dev.numpy(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "Mt, Nt, Kt, kinst",
    [
        (2, 1, 1, 16),  # M = 32 (two M-tiles)
        (1, 2, 1, 16),  # N = 16 (two N-tiles)
        (1, 1, 2, 16),  # K = 32 (two K-tiles, accumulated in place)
        (2, 2, 2, 16),  # 8 tiles
        (4, 1, 1, 16),  # M = 64
        (1, 1, 1, 8),  # m16n8k8 single tile (kHi == 1)
        (1, 1, 3, 8),  # K = 24 -> three k8 tiles (16 does not divide 24)
        (2, 2, 3, 8),  # k8, every dim tiled
    ],
)
def test_cuda_gemm_mma_lowers_tiled(Mt, Nt, Kt, kinst):
    """Every tiling we expect to dispatch must lower, selecting the right mma.

    The k8 cases are the regression guard for the kHi == 1 fragment grouping
    (an extent-1 high-K register group must not be rejected as a thread axis).
    """
    script = _lower(_build_tiled(Mt, Nt, Kt, kinst))["main"].script()
    assert "T.ptx.mma(" in script
    assert f"m16n8k{kinst}" in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "Mt, Nt, Kt, kinst",
    [
        (1, 1, 1, 16),
        (2, 2, 2, 16),
        (4, 1, 1, 16),
        (1, 1, 1, 8),
        (1, 1, 3, 8),
        (2, 2, 3, 8),
    ],
)
def test_cuda_gemm_mma_codegen_issue_count(Mt, Nt, Kt, kinst):
    """Full pipeline (UnrollLoop + CUDA codegen) emits one mma per (Mt, Nt, Kt)
    tile; K-tiles accumulate in place, so D is cleared once per output tile."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tiled(Mt, Nt, Kt, kinst, store=True)}),
            target=target,
            tir_pipeline="tirx",
        )
    src = mod.mod.imports[0].inspect_source()
    assert f"mma.sync.aligned.m16n8k{kinst}" in src
    # mma is emitted as one __device__ helper, invoked once per tile.
    helper = f"ptx_mma_m16n8k{kinst}_row_col"
    assert src.count(helper) - 1 == Mt * Nt * Kt


@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cuda_gemm_mma_lowers_transpose(transpose_A, transpose_B):
    """All four A/B orientations dispatch to the same m16n8k16. transpose only
    describes the input's logical orientation; the .row.col mma is unchanged."""
    script = _lower(_build_transpose(transpose_A, transpose_B))["main"].script()
    assert "T.ptx.mma(" in script
    assert "m16n8k16" in script


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "transpose_A, transpose_B",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cuda_gemm_mma_codegen_transpose(transpose_A, transpose_B):
    """Every orientation codegens to a valid m16n8k16 kernel."""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_transpose(transpose_A, transpose_B, store=True)}),
            target=target,
            tir_pipeline="tirx",
        )
    assert "mma.sync.aligned.m16n8k16" in mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize(
    "a, b, c, d",
    [
        ("float16", "float16", "float16", "float16"),  # f16 accumulate
        ("bfloat16", "float16", "float32", "float32"),  # mixed A/B inputs
        ("float32", "float32", "float32", "float32"),  # f32 (tf32) inputs
        ("int8", "int8", "int32", "int32"),  # integer
    ],
)
def test_cuda_gemm_mma_rejects_unsupported_dtype(a, b, c, d):
    """The table holds only (bf16|f16, same, f32, f32); any other dtype
    signature must decline rather than emit a wrong mma."""
    with pytest.raises(RuntimeError, match="dispatch failed"):
        _lower(_build_dtypes(a, b, c, d))


if __name__ == "__main__":
    tvm.testing.main()
