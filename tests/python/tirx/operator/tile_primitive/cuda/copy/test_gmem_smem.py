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
"""Round-trip tests for the ``gmem_smem`` copy dispatch (synthesized partition).

Pipeline: A_gmem --G2S--> A_smem --S2G--> B_gmem. If either direction is
wrong the round trip leaves B mismatched against A.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout


def _build_kernel(scope, n_threads, shape, dtype):
    s_layout = TileLayout(S[shape])
    full_slices = tuple(slice(0, d) for d in shape)

    if scope == "warpgroup":

        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, shape, dtype)
            B = T.match_buffer(B_ptr, shape, dtype)
            T.device_entry()
            T.cta_id([1])
            T.warpgroup_id([n_threads // 128])
            T.warp_id_in_wg([4])
            T.lane_id([32])
            T.thread_id_in_wg([128])
            T.thread_id([n_threads])
            A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
            Tx.wg.copy(A_smem[full_slices], A[full_slices])
            T.cuda.cta_sync()
            Tx.wg.copy(B[full_slices], A_smem[full_slices])

    elif scope == "warp":

        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, shape, dtype)
            B = T.match_buffer(B_ptr, shape, dtype)
            T.device_entry()
            T.cta_id([1])
            T.lane_id([32])
            T.thread_id([n_threads])
            A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
            Tx.warp.copy(A_smem[full_slices], A[full_slices])
            T.cuda.cta_sync()
            Tx.warp.copy(B[full_slices], A_smem[full_slices])

    elif scope == "cta":

        @T.prim_func
        def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, shape, dtype)
            B = T.match_buffer(B_ptr, shape, dtype)
            T.device_entry()
            T.cta_id([1])
            T.warp_id([n_threads // 32])
            T.lane_id([32])
            T.thread_id([n_threads])
            A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
            Tx.cta.copy(A_smem[full_slices], A[full_slices])
            T.cuda.cta_sync()
            Tx.cta.copy(B[full_slices], A_smem[full_slices])
    else:
        raise ValueError(f"unsupported scope {scope!r}")

    return kernel


# (scope, n_threads, shape) — shape chosen so total / T / vec_len > 1 with at
# least one outer round, and total is divisible by T*vec_len.
TASKS = [
    ("warp", 32, (32, 32)),  # 1024 total, T=32, vec 8 → outer 4
    ("warp", 32, (32, 64)),  # 2048 total, outer 8
    ("warpgroup", 128, (128, 32)),  # 4096, outer 4
    ("warpgroup", 128, (128, 64)),  # 8192, outer 8
    ("warpgroup", 128, (256, 16)),  # 4096, outer 4
    ("cta", 256, (256, 32)),  # 8192, T=256, vec 8 → outer 4
    ("cta", 256, (512, 16)),  # 8192, outer 4
]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "scope,n_threads,shape",
    [pytest.param(*t, id=f"{t[0]}-{t[1]}-{'x'.join(map(str, t[2]))}") for t in TASKS],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "uint8"])
def test_gmem_smem_roundtrip(scope, n_threads, shape, dtype):
    kernel = _build_kernel(scope, n_threads, shape, dtype)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros(shape, dtype=np_dtype)

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), A_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Migrated from test_copy_sync.py: sync G↔S copy via the user-facing
# Tx.copy() (which dispatches to gmem_smem).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        # A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32]
        (
            (128, 32),
            (128, 32),
            ((0, 128), (0, 32)),
            32,
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
        ),
        # A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64]
        (
            (64, 64),
            (32, 32),
            ((32, 64), (32, 64)),
            32,
            TileLayout(S[64, 64]),
            TileLayout(S[64, 64]),
            TileLayout(S[32, 32]),
        ),
        # A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32]
        (
            (4, 32, 32),
            (32, 32),
            ((0, 1), (0, 32), (0, 32)),
            32,
            TileLayout(S[4, 32, 32]),
            TileLayout(S[4, 32, 32]),
            TileLayout(S[32, 32]),
        ),
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),
            (8, 8),
            ((0, 8), (0, 8)),
            32,
            TileLayout(S[16, 16]),
            TileLayout(S[16, 16]),
            TileLayout(S[8, 8]),
        ),
        # A[32:96, 256:512] -> A_smem[0:32, 0:256] -> B[32:96, 256:512] (swizzled)
        (
            (96, 512),
            (32, 256),
            ((16, 48), (256, 512)),
            32,
            TileLayout(S[96, 512]),
            TileLayout(S[96, 512]),
            ComposeLayout(SwizzleLayout(3, 3, 3), TileLayout(S[8, 64]))
            .tile_to((16, 128), (8, 64))
            .tile_to((32, 256), (16, 128)),
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("scope", ["cta", "thread"])
def test_copy_g2s_s2g(task, dtype, scope):
    g_shape, s_shape, g_region, thread_cnt, layoutA, layoutB, layoutS = task

    r_smem = tuple(slice(None) for _ in range(len(s_shape)))
    r_gmem = tuple(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    if scope == "thread":
        thread_cnt = 1

    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        T.device_entry()
        T.cta_id([2])
        T.thread_id([thread_cnt])

        A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)
        # `scope` is parametrized at runtime; select the scope namespace
        # dynamically (T.cta / T.thread) instead of a literal prefix.
        getattr(Tx, scope).copy(A_smem[r_smem], A[r_gmem])
        T.cuda.cta_sync()
        getattr(Tx, scope).copy(B[r_gmem], A_smem[r_smem])

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Regression tests for known correctness gaps in ``align_layouts_gs``.
#
# These are intentionally algorithm-level (no GPU runtime) and currently
# XFAIL; flipping them to passing is the contract for the upcoming swizzle /
# alignment fix.
# ----------------------------------------------------------------------------


def _align(
    g_layout, g_shape, s_layout, s_shape, elem_bits, thread_cnt, g_region=None, s_region=None
):
    from tvm.tirx.cuda.operator.tile_primitive.copy._common import align_layouts_gs

    target = tvm.target.Target("cuda")
    if g_region is None:
        g_region = [(0, d) for d in g_shape]
    if s_region is None:
        s_region = [(0, d) for d in s_shape]
    with target:
        return align_layouts_gs(
            g_layout,
            g_shape,
            g_region,
            s_layout,
            s_shape,
            s_region,
            elem_bits,
            thread_cnt,
        )


@pytest.mark.xfail(
    reason="align_layouts_gs ignores swizzle chunk size; "
    "_extract_tile strips the swizzle wrap before vec_len pick."
)
@pytest.mark.parametrize("per_element,expected_max_vec", [(2, 4), (1, 2), (0, 1)])
def test_swizzled_smem_vec_len_must_fit_chunk(per_element, expected_max_vec):
    """``SwizzleLayout(per_element, ...)`` keeps the bottom ``per_element``
    bits unswizzled. vec must stay within that chunk or it crosses an XOR
    boundary and reads/writes the wrong physical bytes."""
    shape = (32, 32)  # 1024 fp16 elements total
    g_layout = TileLayout(S[shape])
    s_layout = ComposeLayout(SwizzleLayout(per_element, 3, 3), TileLayout(S[shape]))
    _g, _s, vec_len = _align(g_layout, shape, s_layout, shape, elem_bits=16, thread_cnt=32)
    chunk_elems = 1 << per_element
    assert vec_len <= chunk_elems, (
        f"vec_len={vec_len} crosses swizzle chunk size={chunk_elems} "
        f"(SwizzleLayout(per_element={per_element}, ...))"
    )


def test_unaligned_strides_must_clamp_vec_len():
    """G layout with row stride 20 (non-multiple of vec_len=8) → tid=2's
    base offset = 20 elements * 2 bytes = 40 bytes, which is not 16-byte
    aligned for a 128-bit vec ld/st (uint4 reinterpret crashes)."""
    shape = (2, 16)
    # row stride 20 (instead of 16) — leaves 4-elem gap between rows.
    g_layout = TileLayout(S[(2, 16) : (20, 1)])
    s_layout = TileLayout(S[(2, 16) : (20, 1)])
    _g, s_p, vec_len = _align(g_layout, shape, s_layout, shape, elem_bits=16, thread_cnt=4)
    # All non-vec strides must be multiples of vec_len so per-thread / per-round
    # starting offset stays vec-aligned. vec iter is s_p.shard[-1] (always
    # stride=1 by construction).
    for it in s_p.shard[:-1]:
        stride = int(it.stride)
        assert stride % vec_len == 0, (
            f"stride={stride} not a multiple of vec_len={vec_len}; "
            f"per-thread / per-round offset will be misaligned for the vec ld/st"
        )


def test_unaligned_region_offset_must_clamp_vec_len():
    """Slicing the gmem region at a non-vec-aligned column (e.g. col 3 in
    fp16) means the per-thread base offset starts at 3 elements = 6 bytes,
    which is not 16/8/4-byte aligned — vec_len must drop to 1."""
    shape = (4, 16)
    g_layout = TileLayout(S[(4, 32)])  # full buffer is 4x32 fp16
    s_layout = TileLayout(S[(4, 16)])
    # Take cols [3, 19) — start offset 3 (odd for any vec_len > 1 in fp16).
    g_region = [(0, 4), (3, 19)]
    s_region = [(0, 4), (0, 16)]
    _g, _s, vec_len = _align(
        g_layout,
        (4, 32),
        s_layout,
        (4, 16),
        elem_bits=16,
        thread_cnt=4,
        g_region=g_region,
        s_region=s_region,
    )
    assert 3 % vec_len == 0, (
        f"vec_len={vec_len} doesn't divide the region's starting column 3; "
        f"per-thread base offset will be misaligned for the vec ld/st"
    )


def test_swizzled_smem_emit_must_be_swizzle_aware():
    """Codegen-level: emitted S address should go through the SwizzleLayout's
    Apply so the XOR scrambling is honored. Currently emit uses
    ``s_buf.ptr_to([0,..,0]) + linear_offset`` which only matches a
    non-swizzled storage layout."""
    import tvm
    from tvm.script import tirx as T
    from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout

    shape = (128, 32)
    s_layout = ComposeLayout(SwizzleLayout(3, 3, 3), TileLayout(S[shape]))

    @T.prim_func
    def kernel(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float16")
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id_in_wg([128])
        T.thread_id([128])
        A_smem = T.alloc_buffer(shape, "float16", scope="shared", layout=s_layout)
        Tx.wg.copy(A_smem[0:128, 0:32], A[0:128, 0:32])

    # NB: pin sm_90 explicitly — the default cuda target falls back to sm_50
    # when no GPU is detected, which nvcc 13+ rejects. Codegen happens before
    # nvcc; if the whole tvm.compile pipeline fails, we never see the source.
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = "".join(im.inspect_source() for im in compiled.mod.imports)

    # If emit is swizzle-aware, two ways it shows up in the generated
    # CUDA:
    #   1. fallback path emits ``swizzle.apply(linear)``, which lowers
    #      to a ``^`` (XOR) somewhere in the S-offset computation
    #      (typically on a separate ``s_off_ptr[0] = ...`` line, not on
    #      the ``tvm_builtin_pointer_offset`` line itself).
    #   2. fast path precomputes a ``signed_strides[N]`` register array
    #      (one per binary outer iter), so each per-iter offset is a
    #      sum of those strides — fingerprintable by the ``1 - 2 *``
    #      sign-computation idiom emit_init writes.
    # XOR-less code paired with no signed_strides init means swizzle
    # was silently dropped.
    has_xor = "^" in src
    has_signed_strides_init = "1 - 2 *" in src or "(1 - 2 *" in src
    assert has_xor or has_signed_strides_init, (
        "emitted s_ptr address shows no swizzle handling — no XOR (fallback "
        "path) and no signed_strides init (fast path)"
    )


def test_layout_permute_copy_preserves_smem_strides():
    """Regression for the MMA-style K-tiled SMEM layout (``tcgen05_mma_ss_no_tma``):

    ``Tx.copy(A_smem, A)`` where A is plain row-major and A_smem uses the
    K-tiled MMA layout. The two layouts cover the same byte range but map
    ``(i, j)`` to *different* physical offsets:

      A      : ``A[i, j]      → i*K + j``         (row-major)
      A_smem : ``A_smem[i, j] → i*8 + (j//8)*1024 + (j%8)``  (K-tiled)

    Earlier ``align_layouts_gs`` sorted+canonicalized BOTH sides
    independently. A_smem's three iters all chain by stride
    (8*128 == 1024, 1*8 == 8), so ``FuseContiguousShardIters`` collapsed
    A_smem to ``[(8192, 1)]`` — same as A's canonical form. The partition
    then synthesized identical ``s_p`` and ``g_p`` strides, ``apply``
    emitted ``tid*8`` on *both* sides, and the copy treated A_smem as
    row-major. MMA descriptors that re-read A_smem with the K-tiled
    formula then saw 99% wrong elements.

    Fix: ``align_layouts_gs`` only sorts+canonicalizes G. S is grouped by
    G's pre-sort iter extents and permuted by G's stride-desc permutation;
    S's per-iter strides are preserved end-to-end.

    This test asserts the structural property: ``s_p.apply`` for any
    ``tid > 0`` must produce an offset that differs from G's row-major
    ``tid * vec_len`` — proving S kept its K-tiled stride 1024.
    """
    from tvm.tirx import Var as _TirVar
    from tvm.tirx.expr import IntImm as _IntImm

    M, K = 128, 64
    # Plain row-major GMEM.
    g_layout = TileLayout(S[M, K])
    # K-tiled SMEM: 3D shape (M, K//8, 8) with strides (8, M*8, 1) — the
    # MMA descriptor's canonical SWIZZLE=0 layout from
    # tests/.../codegen/test_codegen_blackwell.py::test_tcgen05_mma_ss_no_tma.
    s_layout = TileLayout(S[(M, K // 8, 8) : (8, M * 8, 1)])

    g_p, s_p, vec_len = _align(
        g_layout,
        (M, K),
        s_layout,
        (M, K),
        elem_bits=16,
        thread_cnt=128,
    )

    # vec_len must reach 8 (fp16 → 128-bit vec ld/st).
    assert vec_len == 8, f"expected vec_len=8 for K-tiled fp16 MMA layout, got {vec_len}"

    # S must keep at least one iter with stride 1024 (the K-tile jump
    # between 8-elem columns). After the fix, s_p.shard has 4 iters with
    # strides [128, 8, 1024, 1]; the old (broken) code collapsed to 3
    # iters all matching g_p's row-major strides [1024, 8, 1].
    s_strides = [int(it.stride) for it in s_p.shard]
    assert 1024 in s_strides, (
        f"s_p strides {s_strides} lost the K-tile stride 1024 — "
        f"align_layouts_gs collapsed A_smem to row-major and the copy "
        f"will write A_smem in the wrong layout"
    )

    # Codegen-level check: s_p.apply on (f=0, tid, v=0) must depend on
    # ``tid % 8`` (the K-tile jump), not just ``tid * 8`` (row-major).
    # We pin this by evaluating apply for a couple of concrete tids.
    target = tvm.target.Target("cuda")
    with target:
        apply_shape = [_IntImm("int32", 8), _IntImm("int32", 128), _IntImm("int32", 8)]
        tid_var = _TirVar("tid", "int32")
        s_off_expr = s_p.apply(
            _IntImm("int32", 0),
            tid_var,
            _IntImm("int32", 0),
            shape=apply_shape,
        )["m"]
        g_off_expr = g_p.apply(
            _IntImm("int32", 0),
            tid_var,
            _IntImm("int32", 0),
            shape=apply_shape,
        )["m"]

    # G is row-major: g_off(tid) = tid * 8.
    # S is K-tiled : s_off(tid) = (tid // 8) * 8 + (tid % 8) * 1024.
    # For tid=1 the two MUST differ — they're identical iff S was
    # collapsed to row-major (the regression).
    from tvm.tirx import stmt_functor

    analyzer = tvm.arith.Analyzer()
    s_off_at_1 = analyzer.simplify(
        stmt_functor.substitute(s_off_expr, {tid_var: _IntImm("int32", 1)})
    )
    g_off_at_1 = analyzer.simplify(
        stmt_functor.substitute(g_off_expr, {tid_var: _IntImm("int32", 1)})
    )
    assert int(s_off_at_1) == 1024, (
        f"s_p.apply at tid=1 produced offset {s_off_at_1}, expected 1024 "
        f"(K-tile jump). S was collapsed to row-major somewhere."
    )
    assert int(g_off_at_1) == 8, (
        f"g_p.apply at tid=1 produced offset {g_off_at_1}, expected 8 (row-major)"
    )


# ----------------------------------------------------------------------------
# Fast-path firing test (positive). Pairs with the var_bounds wiring inside
# ``gmem_smem._emit_gmem_smem``.
#
# Setup: warp-scope 32x64 fp16 G2S/S2G with 128b swizzled SMEM. The outer
# iter stride is ``thread_cnt * vec_len = 32 * 8 = 256``, which puts the
# binary-split bj's at {5, 6, 7} — well above the swizzle XOR region (so
# Case 1.D, signed_stride = +T). The (C1) analyzer check
# ``bit_bj(s_off // C) == 0`` needs the placeholder var bounded to
# laneid ∈ [0, 32); the dispatch passes ``var_bounds`` so it can discharge,
# recognizer accepts, and emit lowers to the
# ``base_off + sum_j bit_j(f) · signed_strides[j]`` precomputed form.
# ----------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_gmem_smem_swizzle_fast_path_fires_with_var_bounds():
    """Warp-scope 32x64 fp16 G2S/S2G with 128b swizzled SMEM. Fast path
    must fire: a 3-slot ``v_<n>[]`` signed_strides buffer + bit-select adds
    per outer iter, no per-iter ``swizzle.apply`` XOR splice in the hot path."""
    import re

    swizzle = SwizzleLayout(3, 3, 3)
    shape = (32, 64)
    g_layout = TileLayout(S[shape])
    s_layout = ComposeLayout(swizzle, TileLayout(S[shape]))

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float16", layout=g_layout)
        B = T.match_buffer(B_ptr, shape, "float16", layout=g_layout)
        T.device_entry()
        T.cta_id([1])
        T.lane_id([32])
        T.thread_id([32])
        smem = T.alloc_buffer(shape, "float16", scope="shared", layout=s_layout)
        Tx.warp.copy(smem, A[:, :])
        T.cuda.cta_sync()
        Tx.warp.copy(B[:, :], smem)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    bitsel = re.findall(r"& 1\) \* v_\d+\[", src)
    v_decls = re.findall(r"alignas\(\d+\) int v_\d+\[(\d+)\]", src)
    assert bitsel, (
        "expected fast-path ``(bit & 1) * v_<n>[i]`` adds; if missing, "
        "var_bounds wiring may have regressed"
    )
    assert "3" in v_decls, (
        f"expected at least one 3-slot signed_strides buffer for bjs "
        f"[7, 6, 5]; got decl sizes {v_decls}"
    )

    # Round-trip correctness.
    A_np = np.arange(32 * 64, dtype="float16").reshape(shape)
    B_np = np.zeros(shape, dtype="float16")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        ex(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
