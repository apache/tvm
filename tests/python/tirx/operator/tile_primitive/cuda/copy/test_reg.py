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
"""Round-trip tests for the ``vec_auto`` register copy path.

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
from tvm.testing import env
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


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
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

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    B_np = np.zeros(shape, dtype=np_dtype)
    expected = _expected(shape, dtype)

    def run_and_check():
        dev = tvm.cuda(0)
        B = tvm.runtime.tensor(B_np, dev)
        if non_r_scope == "shared":
            compiled(B)
        else:
            A_np = np.zeros(shape, dtype=np_dtype)
            A = tvm.runtime.tensor(A_np, dev)
            compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_reg_roundtrip_gapped_permuted_storage():
    """Forced reg dispatch addresses sparse register layouts by physical offset."""
    shape = (32, 2, 2)
    r_layout = TileLayout(S[shape : (1 @ laneid, 2, 4)])
    storage = r_layout.storage()
    assert int(storage.size()) == 4
    assert int(storage.span()) == 7
    assert [int(storage.apply(i, shape=[4])["m"]) for i in range(4)] == [0, 4, 2, 6]

    # fmt: off
    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float32")
        B = T.match_buffer(B_ptr, shape, "float32")

        T.device_entry()
        T.cta_id([1])
        T.lane_id([32])
        T.thread_id([32])
        reg = T.alloc_buffer(shape, "float32", scope="local", layout=r_layout)
        Tx.warp.copy(reg, A, dispatch="vec_auto")
        Tx.warp.copy(B, reg, dispatch="vec_auto")
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        compiled = tvm.compile(
            tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx"
        )
    source = compiled.mod.imports[0].inspect_source()
    assert "tvm_builtin_ptx_ld" in source
    assert "tvm_builtin_ptx_st" in source

    rng = np.random.default_rng(0)
    A_np = rng.random(shape, dtype=np.float32)
    B_np = np.zeros(shape, dtype=np.float32)

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), A_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Migrated from test_copy_sync.py: sync G↔L copy via Tx.copy() (L = local =
# per-thread register, so it dispatches to the vec_auto register path).
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
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2l_l2g_vec_load(task, dtype):
    g_shape, l_shape, g_region, thread_cnt, layoutA, layoutB, layoutLocal = task

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

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


# vec_auto reg path must honor cache="nc": a strided [4,4]:(16,1) global->reg
# copy vectorizes to 4x 128b either way, but cache="nc" must emit ld.global.nc.


@T.prim_func
def _nc_strided_reg_copy(src_ptr: T.handle) -> None:
    src = T.match_buffer(src_ptr, (1024,), "int32")
    T.device_entry()
    T.thread_id([128])
    tid = T.thread_id_in_wg([128])
    dst = T.alloc_local((4, 4), "int32")
    if tid == 0:
        # view as (blk, row, warp, j); pick warp=1 -> [4,4]:(16,1)
        blk = src.view(1024 // 64, 4, 4, 4).sub[0, :, 1, :]
        Tx.copy(dst[:, :], blk[:, :], cache="nc")
        for i in T.unroll(4):
            for j in T.unroll(4):
                T.evaluate(dst[i, j])


@T.prim_func
def _plain_strided_reg_copy(src_ptr: T.handle) -> None:
    src = T.match_buffer(src_ptr, (1024,), "int32")
    T.device_entry()
    T.thread_id([128])
    tid = T.thread_id_in_wg([128])
    dst = T.alloc_local((4, 4), "int32")
    if tid == 0:
        blk = src.view(1024 // 64, 4, 4, 4).sub[0, :, 1, :]
        Tx.copy(dst[:, :], blk[:, :])
        for i in T.unroll(4):
            for j in T.unroll(4):
                T.evaluate(dst[i, j])


def test_vec_auto_reg_honors_cache_nc():
    target = tvm.target.Target("cuda")

    def _src(f):
        with target:
            mod = tvm.compile(tvm.IRModule({"main": f}), target=target, tir_pipeline="tirx")
        return mod.mod.imports[0].inspect_source()

    nc_src = _src(_nc_strided_reg_copy)
    plain_src = _src(_plain_strided_reg_copy)

    # cache="nc" -> vectorized 128b ld.global.nc; no cache -> plain 128b ld.
    # Both must vectorize; only the cache qualifier differs.
    assert "ld.global.nc.v4.u32" in nc_src, (
        "vec_auto reg path must vectorize AND honor cache='nc' (emit "
        "ld.global.nc.v4), got:\n"
        + "\n".join(line for line in nc_src.splitlines() if "ld_" in line and "v4" in line)
    )
    assert "ld.global.v4.u32" in plain_src, (
        "no cache hint must vectorize to a plain 128b ld (ld.global.v4)"
    )
    assert "ld.global.nc.v4.u32" not in plain_src, "no cache hint must NOT be nc"


@pytest.mark.gpu
def test_reg_copy_linear_shared_hoists_thread_base():
    """A linear R→S copy must keep its per-thread base outside the hot loop."""
    from tvm.tirx.layout import wg_local_layout

    n_threads, width = 128, 64
    shape = (n_threads, width)
    linear_layout = TileLayout(S[shape])

    @T.prim_func
    def kernel(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, "float16", layout=linear_layout)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([n_threads])
        tid = T.thread_id_in_wg([n_threads])
        reg = T.alloc_buffer(shape, "float16", scope="local", layout=wg_local_layout(width))
        smem = T.alloc_buffer(shape, "float16", scope="shared", layout=linear_layout)

        reg_local = reg.local(width)
        for i in T.serial(width):
            reg_local[i] = A[tid, i]
        Tx.wg.copy(smem, reg)
        T.cuda.cta_sync()
        T.evaluate(smem[tid, 0])

    target = tvm.target.Target("cuda")
    with target:
        ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    base_assignment = "s_base_ptr[0] ="
    loop = "for (int f = 0; f < 8; ++f)"
    assert base_assignment in src and loop in src
    assert src.index(base_assignment) < src.index(loop)
    assert "(s_base_ptr[0] + ds_ptr[0])" in src
    assert "s_off_ptr" not in src


@pytest.mark.gpu
def test_reg_copy_wg_local_to_swizzled_shared_uses_structured_compose_apply():
    """Regression: R→S copy where R has a ``wg_local_layout`` (thread iter
    ``1 @ tid_in_wg``) must pick the widest vec PTX ``st.shared.v4`` and lower
    its synthetic TileLayout through structured ComposeLayout apply.

    Two distinct properties are covered:

    (1) ``_choose_vec_len`` used to include R-side thread-iter strides in
    its alignment check. ``wg_local_layout``'s thread iter has stride 1;
    a vec=8 (16-byte) alignment check on ``1 % 8 != 0`` would reject
    every wider variant and fall to scalar ``copy_16b``. Thread-axis
    strides are partition-coord (virtual), not storage-physical, so they
    must be excluded.

    (2) The hot-loop address must be the direct P/XOR-low/ADD-high chain,
    without full-address quotient/mod decomposition.
    """
    from tvm.tirx.layout import ComposeLayout, wg_local_layout

    N_THREADS, EPI_N = 128, 64
    g_shape = (N_THREADS, EPI_N)
    g_layout = TileLayout(S[g_shape])
    # 128b swizzle on the SMEM side (per_element=3 ⇒ 8 fp16 atom width).
    smem_layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))

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

    # (1) Widest variant: 8 fp16 elements per call (16 bytes → v4.u32 st).
    assert 'asm volatile("st.' in src, (
        "expected PTX st in generated CUDA, alignment check fell back to a narrower variant"
    )
    assert "st.shared.v4" in src, "expected 128b vector store (st.shared.v4.u32)"
    assert "tvm_builtin_copy_" not in src, (
        "copy_xxb helpers appeared — reg dispatch should use PTX ld/st only"
    )
    # (2) Structured address fingerprint: tid contributes one atom-aligned
    # add, while the bounded outer coordinate is XORed with the phase.
    s_off_lines = [
        line
        for line in src.splitlines()
        if line.strip().startswith("s_off_ptr") and "[0] =" in line
    ]
    assert len(s_off_lines) == 1
    assert "^" in s_off_lines[0]
    assert "* 64" in s_off_lines[0] and "f * 8" in s_off_lines[0]
    assert "/" not in s_off_lines[0] and "%" not in s_off_lines[0], (
        "structured hot-loop offset must not contain full quotient/mod decomposition"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_st_from_src_f32_vector_preserves_values():
    """A vector store of f32 registers must preserve the values."""

    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_local((4,), "float32")
        out = T.alloc_local((4,), "float32")
        for i in range(4):
            reg[i] = T.cast(i + 1, "float32")
        T.ptx.st.shared.v4.f32(smem.ptr_to([0]), reg[0], reg[1], reg[2], reg[3])
        T.ptx.ld.shared.v4.f32(out[0], out[1], out[2], out[3], smem.ptr_to([0]))
        for i in range(4):
            B[i] = out[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "st.shared.v4.f32" in src
    # The values stay floats: each lane binds "f", not a reinterpreted "r".
    assert '"f"(__value0)' in src

    dev = tvm.cuda(0)
    out = tvm.runtime.tensor(np.zeros((4,), dtype="float32"), dev)
    ex(out)
    np.testing.assert_equal(out.numpy(), np.array([1, 2, 3, 4], dtype="float32"))


def test_copy_fallback_handles_scalar_regions():
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (1,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        src = T.alloc_local((1,), "float32")
        dst = T.alloc_local((4,), "float32")
        src[0] = T.cast(7, "float32")
        Tx.copy(dst[2:3], src[:])
        B[0] = dst[2]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "dst_ptr[2] = src_ptr[0];" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "variant,dtype,n_elements,expected_st,expected_ld",
    [
        ("vec_16b", "uint16", 1, "st.shared.u16", "ld.shared.u16"),
        ("vec_32b", "uint32", 1, "st.shared.u32", "ld.shared.u32"),
        ("vec_64b", "uint32", 2, "st.shared.v2.u32", "ld.shared.v2.u32"),
        ("vec_128b", "uint32", 4, "st.shared.v4.u32", "ld.shared.v4.u32"),
    ],
)
def test_copy_forced_vec_width_codegen(variant, dtype, n_elements, expected_st, expected_ld):
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (n_elements,), dtype)
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((n_elements,), dtype, scope="shared")
        reg = T.alloc_local((n_elements,), dtype)
        out = T.alloc_local((n_elements,), dtype)
        for i in range(n_elements):
            reg[i] = T.cast(i + 1, dtype)
        Tx.copy(smem[:], reg[:], dispatch=variant)
        T.cuda.cta_sync()
        Tx.copy(out[:], smem[:], dispatch=variant)
        for i in range(n_elements):
            B[i] = out[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert expected_st in src
    assert expected_ld in src

    dev = tvm.cuda(0)
    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    out = tvm.runtime.tensor(np.zeros((n_elements,), dtype=np_dtype), dev)
    ex(out)
    np.testing.assert_equal(out.numpy(), np.arange(1, n_elements + 1, dtype=np_dtype))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_forced_vec_dynamic_swizzled_shared_uses_vector_ptx():
    from tvm.tirx.layout import ComposeLayout

    smem_layout = ComposeLayout(2, 3, 3, TileLayout(S[(64, 8, 32) : (32, 2048, 1)]))

    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (128, 4), "float32")
        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([128])
        smem = T.alloc_buffer((64, 256), "float32", scope="shared", layout=smem_layout)
        reg = T.alloc_local((4,), "float32")
        out = T.alloc_local((4,), "float32")
        for i in range(4):
            reg[i] = T.cast(tid * 16 + i + 1, "float32")
        row: T.let = tid % 64
        col: T.let = (tid // 64) * 4
        Tx.copy(smem[row, col : col + 4], reg[:], dispatch="vec_128b")
        T.cuda.cta_sync()
        Tx.copy(out[:], smem[row, col : col + 4], dispatch="vec_128b")
        for i in range(4):
            B[tid, i] = out[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "copy/fallback" not in src
    assert "st.shared.v4.u32" in src
    assert "ld.shared.v4.u32" in src

    dev = tvm.cuda(0)
    out = tvm.runtime.tensor(np.zeros((128, 4), dtype="float32"), dev)
    ex(out)
    expected = np.array([[tid * 16 + i + 1 for i in range(4)] for tid in range(128)], "float32")
    np.testing.assert_equal(out.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_explicit_vec_auto_uses_auto_family():
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_buffer((4,), "float32", scope="local", layout=TileLayout(S[4]))
        out = T.alloc_buffer((4,), "float32", scope="local", layout=TileLayout(S[4]))
        for i in range(4):
            reg[i] = T.cast(i + 1, "float32")
        Tx.copy(smem[:], reg[:], dispatch="vec_auto")
        T.cuda.cta_sync()
        Tx.copy(out[:], smem[:], dispatch="vec_auto")
        for i in range(4):
            B[i] = out[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "tvm_builtin_copy_" not in src
    assert "st.shared.v4.u32" in src
    assert "ld.shared.v4.u32" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("dispatch", ["reg", "gmem_smem"])
def test_copy_old_dispatch_names_are_not_registered(dispatch):
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_local((4,), "float32")
        Tx.copy(smem[:], reg[:], dispatch=dispatch)
        B[0] = T.cast(0, "float32")

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match=f"no variant named '{dispatch}' is registered"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_forced_vec_rejects_size_mismatch():
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.thread_id([1])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_local((4,), "float32")
        Tx.copy(smem[:], reg[:], dispatch="vec_64b")
        B[0] = T.cast(0, "float32")

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="src region does not contain exactly 2 elements"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_forced_vec_rejects_non_thread_scope():
    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.lane_id([32])
        T.thread_id([32])
        smem = T.alloc_buffer((4,), "float32", scope="shared")
        reg = T.alloc_buffer((4,), "float32", scope="local", layout=TileLayout(S[4]))
        Tx.warp.copy(smem[:], reg[:], dispatch="vec_128b")
        B[0] = T.cast(0, "float32")

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="expected thread exec_scope"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


def test_reg_swizzle_chunk_caps_vector_width():
    from tvm.backend.cuda.tile_primitive.copy.vec_auto_reg import _choose_vec_len

    atoms = [(8, 1, 1)]
    tile = TileLayout(S[8:1])
    assert _choose_vec_len(16, atoms, tile, tile) == 8
    assert _choose_vec_len(16, atoms, tile, tile, max_vec_len=4) == 4


def _eval_const_layout_expr(expr, values):
    node_type = type(expr).__name__
    if node_type == "IntImm":
        return int(expr.value)
    if node_type == "Var":
        return values[expr]
    if node_type in ("Add", "Sub", "Mul", "FloorDiv", "FloorMod"):
        lhs = _eval_const_layout_expr(expr.a, values)
        rhs = _eval_const_layout_expr(expr.b, values)
        if node_type == "Add":
            return lhs + rhs
        if node_type == "Sub":
            return lhs - rhs
        if node_type == "Mul":
            return lhs * rhs
        if node_type == "FloorDiv":
            return lhs // rhs
        return lhs % rhs
    if node_type == "Cast":
        return _eval_const_layout_expr(expr.value, values)
    if node_type == "Call":
        args = [_eval_const_layout_expr(arg, values) for arg in expr.args]
        op_name = str(expr.op.name)
        if op_name == "tirx.bitwise_xor":
            return args[0] ^ args[1]
        if op_name == "tirx.bitwise_and":
            return args[0] & args[1]
        if op_name == "tirx.shift_left":
            return args[0] << args[1]
        if op_name == "tirx.shift_right":
            return args[0] >> args[1]
        raise AssertionError(f"Cannot evaluate call {op_name}")
    raise AssertionError(f"Cannot evaluate node type {node_type}")


@pytest.mark.parametrize("case", ["wg", "wg_slice", "tcgen05"])
def test_reg_synthetic_tile_matches_thread_base_plus_outer_delta(case):
    from tvm.arith import Analyzer
    from tvm.backend.cuda.tile_primitive.copy.vec_auto_reg import (
        _build_atoms,
        _build_s_apply_layout,
        _choose_vec_len,
        _make_thread_placeholders,
        _outer_const_offsets,
        _s_thread_contributions,
        _split_atoms_for_vec,
        _split_thread_loop,
        align_layouts_raw,
    )
    from tvm.tirx.exec_scope import ExecScope
    from tvm.tirx.layout import ComposeLayout, wg_local_layout
    from tvm.tirx.operator.tile_primitive import DispatchContext

    if case in ("wg", "wg_slice"):
        shape = [128, 64]
        region = [(0, 128), (0, 64) if case == "wg" else (8, 40)]
        r_layout = wg_local_layout(64)
        s_layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        elem_bits = 16
        expected_thread_extents = [128]
    else:
        from tvm.tirx.cuda.tile_primitive.tma_utils import mma_shared_layout
        from tvm.tirx.layout import tcgen05_atom_layout

        shape = [64, 64]
        region = [(0, 64), (0, 64)]
        r_layout = tcgen05_atom_layout("16x256b", (64, 64), "float32")
        s_layout = mma_shared_layout("float32", 3, (64, 64))
        elem_bits = 32
        expected_thread_extents = [4, 8, 4]

    target = tvm.target.Target("cuda")
    with target:
        r_p, s_p, s_seps, r_perm = align_layouts_raw(
            r_layout.slice(shape, region), s_layout.slice(shape, region), region
        )
    r_iters, s_groups = _split_thread_loop(r_perm, s_p, s_seps)
    atoms = _build_atoms(r_iters, s_groups)
    max_vec_len = 1 << int(s_layout.per_element)
    vec_len = _choose_vec_len(elem_bits, atoms, r_p, s_p, max_vec_len)
    outer = _split_atoms_for_vec(atoms, vec_len)
    placeholders = _make_thread_placeholders(r_p)
    sctx = DispatchContext(target, ExecScope("warpgroup"), {}, {}, scope_kind="warpgroup")
    s_apply_layout, thread_coords, apply_shape = _build_s_apply_layout(
        s_layout, r_p, s_p, outer, placeholders, sctx
    )

    assert isinstance(s_apply_layout, ComposeLayout)
    assert [int(it.extent) for it in s_apply_layout.tile_layout.shard[: len(thread_coords)]] == (
        expected_thread_extents
    )

    old_base = tvm.tirx.IntImm("int32", 0)
    for coord, stride in _s_thread_contributions(r_p, s_p, placeholders):
        old_base = old_base + coord * stride
    for value in s_p.offset.values():
        old_base = old_base + value

    period = 1 << (int(s_layout.per_element) + int(s_layout.swizzle_len) + int(s_layout.atom_len))
    bare_swizzle = ComposeLayout(
        int(s_layout.per_element),
        int(s_layout.swizzle_len),
        int(s_layout.atom_len),
        TileLayout(S[(period,)]),
        bool(s_layout.swizzle_inner),
    )
    placeholder = next(iter(placeholders.values()))
    analyzer = Analyzer()
    for tid in range(128):
        value_map = {placeholder: tvm.tirx.IntImm("int32", tid)}
        for f in range(int(apply_shape[-1])):
            ds, _dr = _outer_const_offsets(outer, f)
            old_linear = old_base + ds
            synthetic_linear = s_apply_layout.tile_layout.apply(
                *thread_coords, f, shape=apply_shape
            )["m"]
            structured_swizzle = s_apply_layout.apply(*thread_coords, f, shape=apply_shape)["m"]
            naive_swizzle = bare_swizzle.apply(old_linear)["m"]
            assert int(
                analyzer.simplify(tvm.tirx.stmt_functor.substitute(synthetic_linear, value_map))
            ) == int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(old_linear, value_map)))
            assert _eval_const_layout_expr(
                structured_swizzle, value_map
            ) == _eval_const_layout_expr(naive_swizzle, value_map)


# --- tcgen05 D epilogue deposit (tf32_hc_prenorm_gemm) -----------------------
# Production op: ``Tx.warpgroup.copy(smem_cd_mma, d_reg)`` after ``tcgen05.ld``
# pulls the M=64 accumulator fragment from TMEM into ``d_reg``, then deposits it
# into 128B-swizzled MMA SMEM for the subsequent TMA store to gmem D.
_TCGEN05_D_ATOM = "16x256b"
_TCGEN05_D_SHAPE = (64, 64)
_TCGEN05_D_DTYPE = "float32"
_TCGEN05_D_SWIZZLE = 3  # SwizzleMode 128B → mma_shared_layout(..., 3, shape)
_TCGEN05_D_SLICE = (slice(0, 64), slice(0, 64))


def _tcgen05_d_epilogue_layouts():
    from tvm.tirx.cuda.tile_primitive.tma_utils import mma_shared_layout
    from tvm.tirx.layout import tcgen05_atom_layout

    m, n = _TCGEN05_D_SHAPE
    reg_layout = tcgen05_atom_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    return m, n, reg_layout, smem_layout


def _build_tcgen05_d_epilogue_deposit():
    """``Tx.wg.copy(smem[slice], d_reg[slice])``: R (tcgen05 atom) → S (128B swizzle)."""
    from tvm.tirx.cuda.tile_primitive.tma_utils import mma_shared_layout
    from tvm.tirx.layout import tcgen05_atom_layout

    m, n = _TCGEN05_D_SHAPE
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    reg_layout = tcgen05_atom_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    sl_m, sl_n = _TCGEN05_D_SLICE

    @T.prim_func
    def deposit(
        d_reg: T.Buffer((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout),
    ) -> None:
        smem_cd_mma = T.alloc_buffer((m, n), _TCGEN05_D_DTYPE, scope="shared", layout=smem_layout)
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id_in_wg([128])
        Tx.wg.copy(smem_cd_mma[sl_m, sl_n], d_reg[sl_m, sl_n])

    return deposit


def _compile_tcgen05_d_epilogue_deposit():
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": _build_tcgen05_d_epilogue_deposit()})
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


def test_reg_copy_tcgen05_d_epilogue_deposit_layout_pairing():
    """Pre-fix bug: canonical ``r_p`` collapses atom ``m`` groups and drops S pairings.

    Copy: ``Tx.wg.copy(smem_cd_mma[0:64,0:64], d_reg[0:64,0:64])`` (R→S).
    ``d_reg``: ``(64,64)`` fp32 ``tcgen05_atom_layout("16x256b", ...)``.
    ``smem_cd_mma``: ``(64,64)`` fp32 ``mma_shared_layout(..., swizzle=128B)``.
    """
    from tvm.backend.cuda.tile_primitive.copy.vec_auto_reg import (
        _split_thread_loop,
        align_layouts_raw,
    )
    from tvm.tirx.exec_scope import ExecScope
    from tvm.tirx.operator.tile_primitive import DispatchContext

    m, n, reg_layout, smem_layout = _tcgen05_d_epilogue_layouts()
    region = [(0, m), (0, n)]
    sctx = DispatchContext(
        tvm.target.Target("cuda"), ExecScope("warpgroup"), {}, {}, scope_kind="warpgroup"
    )
    with sctx.target:
        r_sliced = reg_layout.slice([m, n], region)
        s_sliced = smem_layout.slice([m, n], region)
        r_p, s_p, s_seps, r_perm = align_layouts_raw(r_sliced, s_sliced, region)

    r_iters, s_groups = _split_thread_loop(r_perm, s_p, s_seps)
    r_iters_bug, s_groups_bug = _split_thread_loop(r_p, s_p, s_seps)
    mem_extents = [int(it.extent) for it in r_iters]
    bug_extents = [int(it.extent) for it in r_iters_bug]

    # Fixed path: 3 register m-groups stay 1:1 with 3 S-side groups.
    assert mem_extents == [8, 2, 2]
    assert len(r_iters) == len(s_groups) == 3
    # Pre-fix path (what _split_thread_loop used to take): 1 fused m-iter, only
    # the first S group is paired — the other two are silently dropped.
    assert bug_extents == [32]
    assert len(r_iters_bug) == len(s_groups_bug) == 1


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_reg_copy_tcgen05_d_epilogue_deposit_codegen():
    """``reg`` dispatch lowers the D epilogue deposit; pre-fix loop only covered half.

    Without the ``r_perm`` fix the emitted outer loop ran ``f < 8`` (one fused
    register group) instead of ``f < 16`` (three atom m-groups x vec tail), and
    the swizzled SMEM stores landed in the wrong (row, col) slots.
    """
    import re

    ex = _compile_tcgen05_d_epilogue_deposit()
    src = ex.mod.imports[0].inspect_source()

    assert "copy/fallback" not in src, "vec_auto register path must not fall back to scalar copy"
    assert "tvm_builtin_copy_" not in src, "vec_auto register path should emit PTX ld/st only"
    assert 'asm volatile("st.' in src
    assert "st.shared.v2.u32" in src, "fp32 vec=2 → 8B shared store per outer iter"

    loop = re.search(r"for \(int f = 0; f < (\d+)", src)
    assert loop is not None, "expected reg copy outer loop in generated CUDA"
    assert loop.group(1) == "16", (
        f"fixed pairing emits 16 outer stores (pre-fix bug collapsed to 8); got f < {loop.group(1)}"
    )


def _tcgen05_16x256b_row_col(tid_wg: T.int32, lane: T.int32, reg_idx: T.int32):
    """Map ``(tid_in_wg, reg)`` → logical ``(row, col)`` for ``.16x256b`` fp32 atom."""
    t0 = lane & T.int32(3)
    t1 = lane >> 2
    v0p = reg_idx & T.int32(1)
    va = (reg_idx >> 1) & T.int32(1)
    vb = reg_idx >> 2
    wid = tid_wg >> 5
    row = t1 + T.int32(8) * va + T.int32(16) * wid
    col = v0p + T.int32(2) * t0 + T.int32(8) * vb
    return row, col


def _build_tcgen05_d_epilogue_deposit_roundtrip():
    """Fill ``d_reg``, R→S deposit, S→R reload, dump via ``.local()`` to gmem."""
    from tvm.tirx.cuda.tile_primitive.tma_utils import mma_shared_layout
    from tvm.tirx.layout import tcgen05_atom_layout

    m, n = _TCGEN05_D_SHAPE
    regs_per_thread = 32  # ``.16x256b.x8`` fp32: 4 regs/slot x rep=8
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    reg_layout = tcgen05_atom_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    sl_m, sl_n = _TCGEN05_D_SLICE

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (m, n), _TCGEN05_D_DTYPE)
        B = T.match_buffer(B_ptr, (m, n), _TCGEN05_D_DTYPE)
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_wg = T.thread_id_in_wg([128])
        lane = T.lane_id([32])
        d_reg = T.alloc_buffer((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout)
        d_reg_out = T.alloc_buffer((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout)
        smem_cd_mma = T.alloc_buffer((m, n), _TCGEN05_D_DTYPE, scope="shared", layout=smem_layout)
        reg_in = d_reg.local(regs_per_thread)
        reg_out = d_reg_out.local(regs_per_thread)
        for r in T.serial(regs_per_thread):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            reg_in[r] = A[row, col]
        Tx.wg.copy(smem_cd_mma[sl_m, sl_n], d_reg[sl_m, sl_n])
        T.cuda.cta_sync()
        Tx.wg.copy(d_reg_out[sl_m, sl_n], smem_cd_mma[sl_m, sl_n])
        for r in T.serial(regs_per_thread):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            B[row, col] = reg_out[r]

    return kernel


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_reg_copy_tcgen05_d_epilogue_deposit_gpu():
    """GPU: R→S deposit + S→R reload must preserve the Layout-F register tile.

    Host fills gmem ``A[row,col]=row*100+col``; each thread scatters into ``d_reg``
    via the ``.16x256b`` (tid,reg)→(row,col) map, runs production
    ``Tx.wg.copy(smem_cd_mma, d_reg)`` then the inverse
    ``Tx.wg.copy(d_reg_out, smem_cd_mma)``, and dumps ``d_reg_out`` back to gmem
    ``B`` through ``.local()``. Pre-fix pairing dropped 2/3 of the S groups —
    ``max|B-A|`` was hundreds, not 0.
    """
    m, n = _TCGEN05_D_SHAPE
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tcgen05_d_epilogue_deposit_roundtrip()}),
            target=target,
            tir_pipeline="tirx",
        )

    rows = np.arange(m, dtype=np.int32)[:, None]
    cols = np.arange(n, dtype=np.int32)[None, :]
    a_np = (rows * 100 + cols).astype(np.float32)
    b_np = np.zeros((m, n), dtype=np.float32)

    def run_and_check():
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(b_np, dev)
        mod(a, b)
        np.testing.assert_allclose(b.numpy(), a_np, rtol=0, atol=0)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
