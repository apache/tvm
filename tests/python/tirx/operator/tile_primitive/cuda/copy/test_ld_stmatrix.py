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
"""Round-trip tests for the ``ldstmatrix`` copy dispatch.

Pipeline:
  ld direction: A_gmem → A_smem (per-thread init) → R_local (T.copy dispatch
                under test) → B_gmem (per-thread write).
  st direction: A_gmem → R_local (per-thread init) → A_smem (T.copy dispatch
                under test) → B_gmem (per-thread write).

Both directions must round-trip ``A == B``. Layout strides are constructed
so that:
  - trans=False S layout matches step-9's row-major spec (8→p, 4→2, num→q, 2→1).
  - trans=True  S layout matches step-9's col-major spec (8→1, 4→2p, num→q, 2→p).

Uniform shape ``(*scope_outer, 8, 4, num, 2)`` is used for every num (including
num=1, which gets an extent-1 placeholder for the num atom).
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout, laneid, tid_in_wg, tx


def _compile_src(kernel):
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": kernel})
    with target:
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")
    return compiled, compiled.mod.imports[0].inspect_source()


# ---------------------------------------------------------------------------
# Layout builders.
# ---------------------------------------------------------------------------
def _r_layout_warp(num):
    return TileLayout(S[(8, 4, num, 2) : (4 @ laneid, 1 @ laneid, 2, 1)])


def _r_layout_warpgroup(num):
    return TileLayout(S[(4, 8, 4, num, 2) : (32 @ tid_in_wg, 4 @ tid_in_wg, 1 @ tid_in_wg, 2, 1)])


def _r_layout_cta(num):
    return TileLayout(S[(4, 8, 4, num, 2) : (32 @ tx, 4 @ tx, 1 @ tx, 2, 1)])


def _s_layout_warp(num, trans):
    if not trans:
        return TileLayout(S[(8, 4, num, 2) : (num * 8, 2, 8, 1)])
    return TileLayout(S[(8, 4, num, 2) : (1, 2 * num * 8, 8, num * 8)])


def _s_layout_warpgroup_or_cta(num, trans):
    if not trans:
        return TileLayout(S[(4, 8, 4, num, 2) : (64 * num, num * 8, 2, 8, 1)])
    return TileLayout(S[(4, 8, 4, num, 2) : (64 * num, 1, 16 * num, 8, 8 * num)])


# 128b swizzle for fp16 (p=3 ⇒ 8 fp16 chunk; sw=at=3 ⇒ 8-row swizzle period).
_SWIZZLE_128B = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)


def _maybe_wrap_swizzle(tile_layout, enable: bool):
    if not enable:
        return tile_layout
    return ComposeLayout(_SWIZZLE_128B, tile_layout)


# ---------------------------------------------------------------------------
# Warp scope kernel builder.
# ---------------------------------------------------------------------------
def _build_warp_kernel(num, direction, trans, swizzle=False):
    r_layout = _r_layout_warp(num)
    s_layout = _maybe_wrap_swizzle(_s_layout_warp(num, trans), swizzle)
    s_shape = (8, 4, num, 2)
    full = (slice(0, 8), slice(0, 4), slice(0, num), slice(0, 2))
    M, N = 8, num * 8

    def _coord(row, cp, t, w):
        # Map per-thread layout coord (row, cp, t, w) to gmem (row, col).
        if not trans:
            return row, t * 8 + cp * 2 + w
        return cp * 2 + w, row + t * 8

    # fmt: off
    if direction == "ld":
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.lane_id([32])
            tid = T.thread_id([32])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            row = tid // 4
            cp = tid % 4
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(row, cp, t, w)
                    A_smem[row, cp, t, w] = A[gr, gc]
            T.cuda.cta_sync()
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            Tx.warp.copy(R_local[full], A_smem[full])
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(row, cp, t, w)
                    B[gr, gc] = r_view[t * 2 + w]
    else:  # direction == "st"
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.lane_id([32])
            tid = T.thread_id([32])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            row = tid // 4
            cp = tid % 4
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(row, cp, t, w)
                    r_view[t * 2 + w] = A[gr, gc]
            Tx.warp.copy(A_smem[full], R_local[full])
            T.cuda.cta_sync()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(row, cp, t, w)
                    B[gr, gc] = A_smem[row, cp, t, w]
    # fmt: on
    return kernel, (M, N)


# ---------------------------------------------------------------------------
# Warpgroup scope kernel builder. 4 warps stacked vertically.
# ---------------------------------------------------------------------------
def _build_warpgroup_kernel(num, direction, trans, swizzle=False):
    r_layout = _r_layout_warpgroup(num)
    s_layout = _maybe_wrap_swizzle(_s_layout_warpgroup_or_cta(num, trans), swizzle)
    s_shape = (4, 8, 4, num, 2)
    full = (slice(0, 4), slice(0, 8), slice(0, 4), slice(0, num), slice(0, 2))
    M, N = 32, num * 8

    def _coord(wid, row, cp, t, w):
        if not trans:
            return wid * 8 + row, t * 8 + cp * 2 + w
        return wid * 8 + cp * 2 + w, row + t * 8

    # fmt: off
    if direction == "ld":
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.warpgroup_id([1])
            T.warp_id_in_wg([4])
            T.lane_id([32])
            T.thread_id_in_wg([128])
            tid = T.thread_id([128])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            wid = tid // 32
            lid = tid % 32
            row = lid // 4
            cp = lid % 4
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    A_smem[wid, row, cp, t, w] = A[gr, gc]
            T.cuda.cta_sync()
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            Tx.wg.copy(R_local[full], A_smem[full])
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    B[gr, gc] = r_view[t * 2 + w]
    else:
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.warpgroup_id([1])
            T.warp_id_in_wg([4])
            T.lane_id([32])
            T.thread_id_in_wg([128])
            tid = T.thread_id([128])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            wid = tid // 32
            lid = tid % 32
            row = lid // 4
            cp = lid % 4
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    r_view[t * 2 + w] = A[gr, gc]
            Tx.wg.copy(A_smem[full], R_local[full])
            T.cuda.cta_sync()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    B[gr, gc] = A_smem[wid, row, cp, t, w]
    # fmt: on
    return kernel, (M, N)


# ---------------------------------------------------------------------------
# CTA scope kernel builder. Same geometry as warpgroup, but R uses ``tx``.
# ---------------------------------------------------------------------------
def _build_cta_kernel(num, direction, trans, swizzle=False):
    r_layout = _r_layout_cta(num)
    s_layout = _maybe_wrap_swizzle(_s_layout_warpgroup_or_cta(num, trans), swizzle)
    s_shape = (4, 8, 4, num, 2)
    full = (slice(0, 4), slice(0, 8), slice(0, 4), slice(0, num), slice(0, 2))
    M, N = 32, num * 8

    def _coord(wid, row, cp, t, w):
        if not trans:
            return wid * 8 + row, t * 8 + cp * 2 + w
        return wid * 8 + cp * 2 + w, row + t * 8

    # fmt: off
    if direction == "ld":
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.warp_id([4])
            T.lane_id([32])
            tid = T.thread_id([128])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            wid = tid // 32
            lid = tid % 32
            row = lid // 4
            cp = lid % 4
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    A_smem[wid, row, cp, t, w] = A[gr, gc]
            T.cuda.cta_sync()
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            Tx.cta.copy(R_local[full], A_smem[full])
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    B[gr, gc] = r_view[t * 2 + w]
    else:
        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, (M, N), "float16")
            B = T.match_buffer(B_ptr, (M, N), "float16")
            T.device_entry()
            T.cta_id([1])
            T.warp_id([4])
            T.lane_id([32])
            tid = T.thread_id([128])
            A_smem = T.alloc_buffer(s_shape, "float16", scope="shared", layout=s_layout)
            wid = tid // 32
            lid = tid % 32
            row = lid // 4
            cp = lid % 4
            R_local = T.alloc_buffer(s_shape, "float16", scope="local", layout=r_layout)
            r_view = R_local.local()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    r_view[t * 2 + w] = A[gr, gc]
            Tx.cta.copy(A_smem[full], R_local[full])
            T.cuda.cta_sync()
            for t in range(num):
                for w in range(2):
                    gr, gc = _coord(wid, row, cp, t, w)
                    B[gr, gc] = A_smem[wid, row, cp, t, w]
    # fmt: on
    return kernel, (M, N)


_BUILDERS = {
    "warp": _build_warp_kernel,
    "warpgroup": _build_warpgroup_kernel,
    "cta": _build_cta_kernel,
}


@pytest.mark.parametrize("scope", ["warp", "warpgroup", "cta"])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("direction", ["ld", "st"])
@pytest.mark.parametrize("num", [1, 2, 4])
@tvm.testing.requires_cuda_compute_version(9)
def test_ldstmatrix(scope, trans, direction, num):
    kernel, (M, N) = _BUILDERS[scope](num, direction, trans)
    compiled, src = _compile_src(kernel)

    inst = "ldmatrix" if direction == "ld" else "stmatrix"
    trans_inst = ".trans" if trans else ""
    expected = f"{inst}.sync.aligned.m8n8.x{num}{trans_inst}.shared.b16"
    assert expected in src, f"{expected} not emitted; src=\n{src}"

    DEV = tvm.cuda(0)
    A_np = np.arange(M * N, dtype="float16").reshape(M, N)
    B_np = np.zeros((M, N), dtype="float16")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    compiled(A, B)
    np.testing.assert_allclose(B.numpy(), A_np)


# ---------------------------------------------------------------------------
# Swizzled-S round-trip. Verifies the ldstmatrix dispatch's swizzle fast
# path (when recognized) and slow path (fallback) both produce correct
# A == B. The 128b swizzle (p=sw=at=3) is the most common fp16 SMEM
# swizzle; with it the dispatch's per-tile S offset goes through
# ``swizzle.apply`` (or its precomputed signed-stride lowering) per mm.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scope", ["warp", "warpgroup", "cta"])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("direction", ["ld", "st"])
@pytest.mark.parametrize("num", [1, 2, 4])
@tvm.testing.requires_cuda_compute_version(9)
def test_ldstmatrix_swizzle(scope, trans, direction, num):
    kernel, (M, N) = _BUILDERS[scope](num, direction, trans, swizzle=True)
    compiled, src = _compile_src(kernel)

    inst = "ldmatrix" if direction == "ld" else "stmatrix"
    trans_inst = ".trans" if trans else ""
    expected = f"{inst}.sync.aligned.m8n8.x{num}{trans_inst}.shared.b16"
    assert expected in src, f"{expected} not emitted; src=\n{src}"

    DEV = tvm.cuda(0)
    A_np = np.arange(M * N, dtype="float16").reshape(M, N)
    B_np = np.zeros((M, N), dtype="float16")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    compiled(A, B)
    np.testing.assert_allclose(B.numpy(), A_np)


# ---------------------------------------------------------------------------
# Multi-iter outer (m_outer > 1) fast-path round-trip. The existing 128b
# swizzle tests above all have m_outer = 1 (only base_off, no signed_strides
# bits). These two cover the non-trivial outer iter case:
#
#   pow2 case (32x64):  R outermost mem ext = 4 (pow2).
#                       m_outer iters on S 6D seg 3 = [(4, 512), (2, 32)].
#                       Binary split → bjs [7, 6, 2]. All BitIter.
#                       signed_strides buffer has 3 slots.
#
#   linear case (40x64): R outermost mem ext = 5 (non-pow2). Stride lands
#                       the outermost S 6D seg 3 iter at (5, 512). 512 is
#                       exactly 2^(p+at+sw) = swizzle period → Case 1.D pure,
#                       so the LinearIter relaxation accepts it.
#                       outer_iters = [LinearIter(5, 512), BitIter(2, ...)].
#                       Inner BitIter (bj=2 Case 1.A) is the only slot in
#                       signed_strides; the outer LinearIter contributes
#                       ``c * 512`` per mm as a compile-time constant.
# ---------------------------------------------------------------------------
def _build_multi_iter_kernel(outer_ext: int):
    """Warp + R=(outer_ext, 8, 2, 4, 4, 2):(16, 4@laneid, 8, 2, 1@laneid, 1)
    + 333 swizzle on S. Mem strides 16/8/2/1 on extents outer_ext/2/4/2 are
    bijective for outer_ext ∈ {4, 5}: max = (outer_ext-1)*16 + 8 + 6 + 1
    = outer_ext*16 - 1, matching extent product = 16*outer_ext."""
    shape = (outer_ext, 8, 2, 4, 4, 2)
    r_layout = TileLayout(S[shape : (16, 4 @ laneid, 8, 2, 1 @ laneid, 1)])
    s_layout = SwizzleLayout(3, 3, 3)
    full = tuple(slice(0, e) for e in shape)

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float16")
        B = T.match_buffer(B_ptr, shape, "float16")
        T.device_entry()
        T.cta_id([1])
        T.lane_id([32])
        tid = T.thread_id([32])
        A_smem = T.alloc_buffer(shape, "float16", scope="shared", layout=s_layout)
        for a in range(outer_ext):
            for c in range(2):
                for d in range(4):
                    for e in range(2):
                        A_smem[a, tid // 4, c, d, tid % 4, e] = A[a, tid // 4, c, d, tid % 4, e]
        T.cuda.cta_sync()
        R_local = T.alloc_buffer(shape, "float16", scope="local", layout=r_layout)
        Tx.warp.copy(R_local[full], A_smem[full])
        r_view = R_local.local()
        for a in range(outer_ext):
            for c in range(2):
                for d in range(4):
                    for e in range(2):
                        B[a, tid // 4, c, d, tid % 4, e] = r_view[a * 16 + c * 8 + d * 2 + e]

    return kernel, shape


@tvm.testing.requires_cuda_compute_version(9)
def test_ldstmatrix_swizzle_multi_iter_pow2():
    """32x64 fp16 warp; outer m_outer split into multiple BitIters (no
    LinearIter). Fast path must fire with a 3-slot signed_strides buffer."""
    import re

    kernel, shape = _build_multi_iter_kernel(outer_ext=4)
    compiled, src = _compile_src(kernel)
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in src

    # Fast-path fingerprint: 3-slot signed_strides + bit-select uses.
    assert re.search(r"alignas\(\d+\) int v_\d+\[3\]", src), (
        "expected 3-slot signed_strides buffer for bjs [7, 6, 2]"
    )
    bitsel = re.findall(r"& 1\) \* v_\d+\[", src)
    assert bitsel, "fast-path bit-select pattern '& 1) * v_<n>[' missing"

    DEV = tvm.cuda(0)
    n_elem = 1
    for e in shape:
        n_elem *= e
    A_np = np.arange(n_elem, dtype="float16").reshape(shape)
    B_np = np.zeros(shape, dtype="float16")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    compiled(A, B)
    np.testing.assert_allclose(B.numpy(), A_np)


@tvm.testing.requires_cuda_compute_version(9)
def test_ldstmatrix_swizzle_multi_iter_linear():
    """40x64 fp16 warp; outer ext=5 is non-pow2 but stride lands on swizzle
    period (Case 1.D pure) so the LinearIter relaxation fires. Pattern has
    a 1-slot signed_strides (inner BitIter bj=2 Case 1.A); outer iter
    contributes ``c * 512`` per mm as a compile-time constant."""
    import re

    kernel, shape = _build_multi_iter_kernel(outer_ext=5)
    compiled, src = _compile_src(kernel)
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in src

    # Fast-path fingerprint: 1-slot signed_strides (just the inner BitIter).
    assert re.search(r"alignas\(\d+\) int v_\d+\[1\]", src), (
        "expected 1-slot signed_strides buffer (only the inner Case-1.A bj=2)"
    )
    bitsel = re.findall(r"& 1\) \* v_\d+\[", src)
    assert bitsel, "fast-path bit-select pattern missing"

    DEV = tvm.cuda(0)
    n_elem = 1
    for e in shape:
        n_elem *= e
    A_np = np.arange(n_elem, dtype="float16").reshape(shape)
    B_np = np.zeros(shape, dtype="float16")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    compiled(A, B)
    np.testing.assert_allclose(B.numpy(), A_np)
