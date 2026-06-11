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
"""Round-trip tests for the ``reg`` copy dispatch.

R = per-thread local (register). The dispatch handles round-trips between R
and any non-R buffer (``shared*`` or ``global``); ``non_r_scope`` parametrize
toggles which side is exercised.

Self-contained: each thread direct-stores its row into the non-R buffer (no
G2S / G2L dispatch needed because each thread writes its own address), the
dispatch does the inbound copy into R and the outbound copy back, then each
thread reads its row into ``B``. Round-trip mismatch ⇒ at least one direction
is wrong.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, laneid, tid_in_wg, tx


def _r_layout(scope, shape):
    if scope == "warpgroup":
        return TileLayout(S[shape : (1 @ tid_in_wg, 1)])
    if scope == "warp":
        return TileLayout(S[shape : (1 @ laneid, 1)])
    if scope == "cta":
        return TileLayout(S[shape : (1 @ tx, 1)])
    raise ValueError(f"unsupported scope {scope!r}")


def _build_roundtrip_kernel(scope, n_threads, k, dtype, non_r_scope):
    """Build a kernel that round-trips data through R via ``non_r_scope``.

    ``non_r_scope == "shared"``: ``A_smem`` is allocated inside the kernel.
    Kernel signature: ``kernel(B_ptr)``.

    ``non_r_scope == "global"``: a separate gmem ``A`` is the staging area.
    Kernel signature: ``kernel(A_ptr, B_ptr)``.
    """
    shape = (n_threads, k)
    full_slices = (slice(0, n_threads), slice(0, k))
    r_layout = _r_layout(scope, shape)

    if non_r_scope == "shared":
        s_layout = TileLayout(S[shape])

        if scope == "warpgroup":

            @T.prim_func
            def kernel(B_ptr: T.handle) -> None:
                B = T.match_buffer(B_ptr, shape, dtype)
                T.device_entry()
                T.cta_id([1])
                T.warpgroup_id([n_threads // 128])
                T.warp_id_in_wg([4])
                T.lane_id([32])
                T.thread_id_in_wg([128])
                tid = T.thread_id([n_threads])
                A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.wg.copy(R_local[full_slices], A_smem[full_slices])
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.wg.copy(A_smem[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A_smem[tid, kk]

        elif scope == "warp":

            @T.prim_func
            def kernel(B_ptr: T.handle) -> None:
                B = T.match_buffer(B_ptr, shape, dtype)
                T.device_entry()
                T.cta_id([1])
                T.lane_id([32])
                tid = T.thread_id([n_threads])
                A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.warp.copy(R_local[full_slices], A_smem[full_slices])
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.warp.copy(A_smem[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A_smem[tid, kk]

        elif scope == "cta":

            @T.prim_func
            def kernel(B_ptr: T.handle) -> None:
                B = T.match_buffer(B_ptr, shape, dtype)
                T.device_entry()
                T.cta_id([1])
                T.warp_id([n_threads // 32])
                T.lane_id([32])
                tid = T.thread_id([n_threads])
                A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.cta.copy(R_local[full_slices], A_smem[full_slices])
                for kk in range(k):
                    A_smem[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.cta.copy(A_smem[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A_smem[tid, kk]

        return kernel

    if non_r_scope == "global":
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
                tid = T.thread_id([n_threads])
                for kk in range(k):
                    A[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.wg.copy(R_local[full_slices], A[full_slices])
                for kk in range(k):
                    A[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.wg.copy(A[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A[tid, kk]

        elif scope == "warp":

            @T.prim_func
            def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
                A = T.match_buffer(A_ptr, shape, dtype)
                B = T.match_buffer(B_ptr, shape, dtype)
                T.device_entry()
                T.cta_id([1])
                T.lane_id([32])
                tid = T.thread_id([n_threads])
                for kk in range(k):
                    A[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.warp.copy(R_local[full_slices], A[full_slices])
                for kk in range(k):
                    A[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.warp.copy(A[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A[tid, kk]

        elif scope == "cta":

            @T.prim_func
            def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
                A = T.match_buffer(A_ptr, shape, dtype)
                B = T.match_buffer(B_ptr, shape, dtype)
                T.device_entry()
                T.cta_id([1])
                T.warp_id([n_threads // 32])
                T.lane_id([32])
                tid = T.thread_id([n_threads])
                for kk in range(k):
                    A[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
                T.cuda.cta_sync()
                R_local = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
                Tx.cta.copy(R_local[full_slices], A[full_slices])
                for kk in range(k):
                    A[tid, kk] = T.cast(0, dtype)
                T.cuda.cta_sync()
                Tx.cta.copy(A[full_slices], R_local[full_slices])
                T.cuda.cta_sync()
                for kk in range(k):
                    B[tid, kk] = A[tid, kk]

        return kernel

    raise ValueError(f"unsupported non_r_scope {non_r_scope!r}")


def _expected(shape, dtype):
    n, k = shape
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    out = np.empty(shape, dtype=np_dtype)
    for t in range(n):
        for kk in range(k):
            out[t, kk] = (t * 100 + kk + 1) % 256 if dtype == "uint8" else t * 100 + kk + 1
    return out


@pytest.mark.parametrize("non_r_scope", ["shared", "global"])
@pytest.mark.parametrize(
    "scope,n_threads,k",
    [
        ("warpgroup", 128, 16),
        ("warpgroup", 128, 32),
        ("warpgroup", 128, 8),
        ("warp", 32, 8),
        ("warp", 32, 16),
        ("cta", 256, 8),
        ("cta", 256, 16),
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "uint8"])
def test_reg_roundtrip(scope, n_threads, k, dtype, non_r_scope):
    shape = (n_threads, k)
    kernel = _build_roundtrip_kernel(scope, n_threads, k, dtype, non_r_scope)

    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    B_np = np.zeros(shape, dtype=np_dtype)
    B = tvm.runtime.tensor(B_np, dev)
    expected = _expected(shape, dtype)
    if non_r_scope == "shared":
        compiled(B)
    else:
        A_np = np.zeros(shape, dtype=np_dtype)
        A = tvm.runtime.tensor(A_np, dev)
        compiled(A, B)
    np.testing.assert_array_equal(B.numpy(), expected)


# ----------------------------------------------------------------------------
# Migrated from test_copy_sync.py: sync G↔L copy via Tx.copy() (L = local =
# per-thread register, so it dispatches to the reg variant).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        # A[3:4, 8:16, 8:16] -> A_local[0:8, 0:8] -> B[3:4, 8:16, 8:16]
        (
            (4, 16, 16),  # g_shape
            (8, 8),  # l_shape
            ((3, 4), (8, 16), (8, 16)),  # g_region
            1,  # thread_cnt
            TileLayout(S[4, 16, 16]),  # layoutA
            TileLayout(S[4, 16, 16]),  # layoutB
            TileLayout(S[8, 8]),  # layoutLocal
            tvm.cuda(0),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2l_l2g_vec_load(task, dtype):
    g_shape, l_shape, g_region, thread_cnt, layoutA, layoutB, layoutLocal, dev = task

    r_lmem = tuple(slice(None) for _ in range(len(l_shape)))
    r_gmem = tuple(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        T.device_entry()
        T.cta_id([2])
        T.thread_id([thread_cnt])
        A_local = T.alloc_buffer(l_shape, dtype, scope="local", layout=layoutLocal)
        Tx.copy(A_local[r_lmem], A[r_gmem])
        Tx.copy(B[r_gmem], A_local[r_lmem])

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]
        np.testing.assert_allclose(B_ref, B.numpy())


def test_reg_copy_wg_local_to_swizzled_shared_uses_swizzle_fastpath():
    """Regression: R→S copy where R has a ``wg_local_layout`` (thread iter
    ``1 @ tid_in_wg``) must pick the widest vec ``copy_128b`` AND use the
    swizzle fast path (precomputed ``signed_strides`` + per-iter
    bit-select), not the per-iter ``swizzle.apply()`` fallback.

    Two distinct bugs this test guards against:

    (1) ``_choose_vec_len`` used to include R-side thread-iter strides in
    its alignment check. ``wg_local_layout``'s thread iter has stride 1;
    a vec=8 (16-byte) alignment check on ``1 % 8 != 0`` would reject
    every wider variant and fall to scalar ``copy_16b``. Thread-axis
    strides are partition-coord (virtual), not storage-physical, so they
    must be excluded.

    (2) Even at the widest vec, if the outer loop is a runtime serial
    (Python ``range`` doesn't actually unroll in TVMScript) the swizzle
    fast path's per-iter constant-fold can't kick in and the
    ``tvm_builtin_pointer_offset`` swizzle XOR ends up recomputed every
    iteration. Loop must be ``T.unroll``.
    """
    from tvm.tirx.layout import SwizzleLayout, wg_local_layout

    N_THREADS, EPI_N = 128, 64
    g_shape = (N_THREADS, EPI_N)
    g_layout = TileLayout(S[g_shape])
    # 128b swizzle on the SMEM side (per_element=3 ⇒ 8 fp16 atom width).
    smem_layout = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, "float16", layout=g_layout)
        B = T.match_buffer(B_ptr, g_shape, "float16", layout=g_layout)

        T.device_entry()
        T.cta_id([1])
        T.thread_id([N_THREADS])
        tid = T.thread_id_in_wg([N_THREADS])
        reg = T.alloc_buffer(g_shape, "float16", scope="local", layout=wg_local_layout(EPI_N))
        smem = T.alloc_buffer(g_shape, "float16", scope="shared", layout=smem_layout)

        # Populate the per-thread slice via .local() (decomposes the wg
        # thread-axis layout into a per-thread 1D view).
        reg_local = reg.local(EPI_N)
        for i in T.serial(EPI_N):
            reg_local[i] = A[tid, i]
        Tx.wg.copy(smem, reg)
        T.cuda.cta_sync()
        for i in T.serial(EPI_N):
            B[tid, i] = smem[tid, i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    # (1) Widest variant: 8 fp16 elements per call.
    assert "tvm_builtin_copy_128b" in src, (
        "expected copy_128b in generated CUDA, alignment check fell back to a narrower variant"
    )
    assert "tvm_builtin_copy_16b" not in src, (
        "scalar copy_16b appeared — vec=8 was wrongly rejected"
    )
    # (2) Swizzle fast path fingerprint:
    #   * emit_init allocates a size-N int buffer of "signed strides".
    #   * emit_iter_offset uses bit-select * signed-stride: ``(bit) * v[i]``
    #     where ``bit = (f >> M) & 1``.
    # The fallback (per-iter ``swizzle.apply(s_off + ds_per_iter)``) has no
    # such bit-select * signed-stride pattern.
    import re

    bitsel_pattern = re.findall(r"& 1\) \* v_\d+\[", src)
    assert bitsel_pattern, (
        "fast-path bit-select pattern '& 1) * v_<n>[' not found; "
        "looks like emit_iter_offset's fast path didn't fire."
    )


if __name__ == "__main__":
    tvm.testing.main()
