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
"""Native ld/st instructions over register layouts and memory regions.

Each zero-copy recipe exposes (group, repeat, word); it changes neither the
backing allocations nor the physical positions of values.
"""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tirx.layout_fixtures import tcgen05_register_layout
from tvm_tirx.backend.cuda.axis import Axis as CUDAAxis
from tvm_tirx.backend.cuda.axis import laneid, tid_in_wg, tx
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


def _r_layout(scope, shape):
    if scope == "warpgroup":
        return TileLayout(S[shape : (1 @ tid_in_wg, 1)])
    if scope == "warp":
        return TileLayout(S[shape : (1 @ laneid, 1)])
    if scope == "cta":
        return TileLayout(S[shape : (1 @ tx, 1)])
    raise ValueError(f"unsupported scope {scope!r}")


def _build_roundtrip_kernel(scope, n_threads, k, dtype, non_r_scope):
    shape = (n_threads, k)
    r_layout = _r_layout(scope, shape)
    # The caller chooses a native width from this test's fixed row size.
    words = min(4, k * tvm_tirx.DataType(dtype).bits // 32)
    load = f"ld.{non_r_scope}.v{words}.u32"
    store = f"st.{non_r_scope}.v{words}.u32"
    if non_r_scope == "shared":

        @T.function
        def kernel(B: T.Tensor(shape, dtype)) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=n_threads))
            tid = T.cuda.thread_idx("x")
            A_smem = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="shared", layout=TileLayout(S[shape]))]
            )
            R_local = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="local", layout=r_layout)]
            )
            for kk in range(k):
                A_smem[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
            T.cuda.cta_sync()
            T.cuda.tile.apply(
                T.ptx[load],
                _thread_words(R_local, scope, words),
                _thread_words(A_smem, scope, words),
                scope=scope,
            )
            for kk in range(k):
                A_smem[tid, kk] = T.cast(0, dtype)
            T.cuda.cta_sync()
            T.cuda.tile.apply(
                T.ptx[store],
                _thread_words(A_smem, scope, words),
                _thread_words(R_local, scope, words),
                scope=scope,
            )
            T.cuda.cta_sync()
            for kk in range(k):
                B[tid, kk] = A_smem[tid, kk]
    else:

        @T.function
        def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=n_threads))
            tid = T.cuda.thread_idx("x")
            R_local = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="local", layout=r_layout)]
            )
            for kk in range(k):
                A[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
            T.cuda.cta_sync()
            T.cuda.tile.apply(
                T.ptx[load],
                _thread_words(R_local, scope, words),
                _thread_words(A, scope, words),
                scope=scope,
            )
            for kk in range(k):
                A[tid, kk] = T.cast(0, dtype)
            T.cuda.cta_sync()
            T.cuda.tile.apply(
                T.ptx[store],
                _thread_words(A, scope, words),
                _thread_words(R_local, scope, words),
                scope=scope,
            )
            T.cuda.cta_sync()
            for kk in range(k):
                B[tid, kk] = A[tid, kk]

    return kernel


def _thread_words(tensor, scope, words):
    if scope == "warp":
        return tensor.view("uint32").rearrange("t (r v) -> t r v", v=words)
    return tensor.view("uint32").rearrange("(w t) (r v) -> w t r v", t=32, v=words)


def _epilogue_pairs(tensor):
    # lane = row_in_half * 4 + column_pair; each instruction transfers two f32s.
    return tensor.view("uint32").rearrange(
        "(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2
    )


def _count_tile_instructions(kernel, opcode):
    target = tvm_tirx.target.Target("cuda")
    with target:
        ex = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": kernel}), target=target, pipeline="tirx"
        )
    return _native_call_count(ex, opcode)


def _native_call_count(ex, opcode):
    name = "tvm_builtin_ptx_" + opcode.replace("st.", "st.vec.").replace(".", "_")
    return ex.mod.imports[0].inspect_source().count("\n  " + name + "(")


def _expected(shape, dtype):
    n, k = shape
    np_dtype = tvm_tirx.testing.np_dtype_from_str(dtype)
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

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        compiled = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

    np_dtype = tvm_tirx.testing.np_dtype_from_str(dtype)
    B_np = np.zeros(shape, dtype=np_dtype)
    expected = _expected(shape, dtype)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        B = tvm_tirx.runtime.tensor(B_np, dev)
        if non_r_scope == "shared":
            compiled(B)
        else:
            A_np = np.zeros(shape, dtype=np_dtype)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), expected)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_reg_roundtrip_gapped_permuted_storage():
    """Forced reg dispatch addresses sparse register layouts by physical offset."""
    shape = (32, 2, 2)
    r_layout = TileLayout(S[shape : (1 @ laneid, 2, 4)])
    storage = TileLayout(S[(2, 2) : (2, 4)])
    assert int(storage.size()) == 4
    assert int(storage.span()) == 7
    assert [int(storage.apply(i, shape=[4])["m"]) for i in range(4)] == [0, 4, 2, 6]

    # fmt: off
    @T.function
    def kernel(A: T.Tensor(shape, 'float32'), B: T.Tensor(shape, 'float32')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        _lane = T.cuda.lane_id()
        reg = T.alloc_tensor(ty_args=[T.Tensor(shape, "float32", scope="local", layout=r_layout)])
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], reg.view(32, 4), A.view(32, 4), scope="warp")
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], B.view(32, 4), reg.view(32, 4), scope="warp")
        # fmt: on

    target = tvm_tirx.target.Target("cuda")
    with target:
        compiled = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": kernel}), target=target, pipeline="tirx"
        )
    source = compiled.mod.imports[0].inspect_source()
    assert "ld.global.v4.b32" in source
    assert "st.global.v4.b32" in source

    rng = np.random.default_rng(0)
    A_np = rng.random(shape, dtype=np.float32)
    B_np = np.zeros(shape, dtype=np.float32)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        A = tvm_tirx.runtime.tensor(A_np, dev)
        B = tvm_tirx.runtime.tensor(B_np, dev)
        compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), A_np)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Original global/local regions, staged with explicitly selected ld/st widths.
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

    words = 2 if tvm_tirx.DataType(dtype).bits == 8 else 4
    load, store = f"ld.global.v{words}.b32", f"st.global.v{words}.b32"
    r_gmem = tuple(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    @T.function
    def copy_sync(
        A: T.Tensor(g_shape, dtype, layout=layoutA), B: T.Tensor(g_shape, dtype, layout=layoutB)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(thread_cnt,)))
        A_local = T.alloc_tensor(
            ty_args=[T.Tensor(l_shape, dtype, scope="local", layout=layoutLocal)]
        )
        T.cuda.tile.apply(
            T.ptx[load],
            A_local.view("uint32").rearrange("r (q v) -> r q v", v=words),
            A.sub[r_gmem].view(8, 8).view("uint32").rearrange("r (q v) -> r q v", v=words),
        )
        T.cuda.tile.apply(
            T.ptx[store],
            B.sub[r_gmem].view(8, 8).view("uint32").rearrange("r (q v) -> r q v", v=words),
            A_local.view("uint32").rearrange("r (q v) -> r q v", v=words),
        )

    np_dtype = tvm_tirx.testing.np_dtype_from_str(dtype)
    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": copy_sync})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        np.random.seed(0)
        A_np = tvm_tirx.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            B = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


# Both explicit opcodes perform four 128-bit loads on the original strided region.


@T.function
def _nc_strided_reg_copy(src: T.Tensor((1024,), "int32", layout="default")) -> None:
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(128,)))
    tid = T.cuda.thread_in_warpgroup()
    dst = T.alloc_local((4, 4), "int32", layout="default")
    if tid == 0:
        # view as (blk, row, warp, j); pick warp=1 -> [4,4]:(16,1)
        blk = src.view(1024 // 64, 4, 4, 4).sub[0, :, 1, :]
        T.cuda.tile.apply(T.ptx["ld.global.nc.v4.u32"], dst.view("uint32"), blk.view("uint32"))
        for i in T.unroll(4):
            for j in T.unroll(4):
                T.evaluate(dst[i, j])


@T.function
def _plain_strided_reg_copy(src: T.Tensor((1024,), "int32", layout="default")) -> None:
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(128,)))
    tid = T.cuda.thread_in_warpgroup()
    dst = T.alloc_local((4, 4), "int32", layout="default")
    if tid == 0:
        blk = src.view(1024 // 64, 4, 4, 4).sub[0, :, 1, :]
        T.cuda.tile.apply(T.ptx["ld.global.v4.u32"], dst.view("uint32"), blk.view("uint32"))
        for i in T.unroll(4):
            for j in T.unroll(4):
                T.evaluate(dst[i, j])


def test_reg_explicit_cache_nc():
    target = tvm_tirx.target.Target("cuda")

    def _src(f):
        with target:
            mod = tvm_tirx.tirx.compile(
                tvm_tirx.IRModule({"main": f}), target=target, pipeline="tirx"
            )
        return mod.mod.imports[0].inspect_source()

    nc_src = _src(_nc_strided_reg_copy)
    plain_src = _src(_plain_strided_reg_copy)

    # cache="nc" -> vectorized 128b ld.global.nc; no cache -> plain 128b ld.
    # Both must vectorize; only the cache qualifier differs.
    assert "ld.global.nc.v4.u32" in nc_src, (
        "The requested non-coherent opcode must survive lowering (emit "
        "ld.global.nc.v4), got:\n"
        + "\n".join(line for line in nc_src.splitlines() if "ld_" in line and "v4" in line)
    )
    assert "ld.global.v4.u32" in plain_src, "The requested plain opcode must survive lowering"
    assert "ld.global.nc.v4.u32" not in plain_src, "no cache hint must NOT be nc"


@pytest.mark.gpu
def test_reg_copy_linear_shared_explicit_repeats():
    """The original 64 fp16 values per thread require eight 128-bit stores."""

    n_threads, width = 128, 64
    shape = (n_threads, width)
    linear_layout = TileLayout(S[shape])

    @T.function
    def kernel(A: T.Tensor(shape, "float16", layout=linear_layout)) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(n_threads,)))
        tid = T.cuda.thread_in_warpgroup()
        reg = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    shape,
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(128, width) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        smem = T.alloc_tensor(
            ty_args=[T.Tensor(shape, "float16", scope="shared", layout=linear_layout)]
        )

        reg_local = reg.view(width, layout="default")
        for i in T.serial(width):
            reg_local[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["st.shared.v4.u32"],
            _thread_words(smem, "warpgroup", 4),
            _thread_words(reg, "warpgroup", 4),
            scope="warpgroup",
        )
        T.cuda.cta_sync()
        T.evaluate(smem[tid, 0])

    assert _count_tile_instructions(kernel, "st.shared.v4.u32") == 8


@pytest.mark.gpu
def test_reg_copy_wg_local_to_swizzled_shared_uses_structured_compose_apply():
    """The original swizzled allocation accepts explicit 128-bit stores."""
    from tvm_tirx.tirx.layout import ComposeLayout

    N_THREADS, EPI_N = 128, 64
    g_shape = (N_THREADS, EPI_N)
    g_layout = TileLayout(S[g_shape])
    # 128b swizzle on the SMEM side (per_element=3 ⇒ 8 fp16 atom width).
    smem_layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))

    @T.function
    def kernel(
        A: T.Tensor(g_shape, "float16", layout=g_layout),
        B: T.Tensor(g_shape, "float16", layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N_THREADS,)))
        tid = T.cuda.thread_in_warpgroup()
        reg = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    g_shape,
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(128, EPI_N) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        smem = T.alloc_tensor(
            ty_args=[T.Tensor(g_shape, "float16", scope="shared", layout=smem_layout)]
        )

        # Populate the per-thread slice via physical-register (decomposes the wg
        # thread-axis layout into a per-thread 1D view).
        reg_local = reg.view(EPI_N, layout="default")
        for i in T.serial(EPI_N):
            reg_local[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["st.shared.v4.u32"],
            _thread_words(smem, "warpgroup", 4),
            _thread_words(reg, "warpgroup", 4),
            scope="warpgroup",
        )
        T.cuda.cta_sync()
        for i in T.serial(EPI_N):
            B[tid, i] = smem[tid, i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert _native_call_count(ex, "st.shared.v4.u32") == 8
    assert "^" in src
    values = np.arange(np.prod(g_shape), dtype="float32").reshape(g_shape).astype("float16")
    out = tvm_tirx.runtime.tensor(np.zeros(g_shape, dtype="float16"), tvm_tirx.cuda())
    ex(tvm_tirx.runtime.tensor(values, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), values)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_st_from_src_f32_vector_preserves_values():
    """A vector store of f32 registers must preserve the values."""

    @T.function
    def kernel(B: T.Tensor((4,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", layout="default")])
        reg = T.alloc_local((4,), "float32", layout="default")
        out = T.alloc_local((4,), "float32", layout="default")
        for i in range(4):
            reg[i] = T.cast(i + 1, "float32")
        T.ptx.st.shared.v4.f32(smem.ptr_to([0]), reg[0], reg[1], reg[2], reg[3])
        T.ptx.ld.shared.v4.f32(out[0], out[1], out[2], out[3], smem.ptr_to([0]))
        for i in range(4):
            B[i] = out[i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "st.shared.v4.f32" in src
    # The values stay floats: each lane binds "f", not a reinterpreted "r".
    assert '"f"(__value0)' in src

    dev = tvm_tirx.cuda(0)
    out = tvm_tirx.runtime.tensor(np.zeros((4,), dtype="float32"), dev)
    ex(out)
    np.testing.assert_equal(out.numpy(), np.array([1, 2, 3, 4], dtype="float32"))


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
    @T.function
    def kernel(B: T.Tensor((n_elements,), dtype)) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(
            ty_args=[T.Tensor((n_elements,), dtype, scope="shared", layout="default")]
        )
        reg = T.alloc_local((n_elements,), dtype, layout="default")
        out = T.alloc_local((n_elements,), dtype, layout="default")
        for i in range(n_elements):
            reg[i] = T.cast(i + 1, dtype)
        T.cuda.tile.apply(T.ptx[expected_st], smem, reg)
        T.cuda.cta_sync()
        T.cuda.tile.apply(T.ptx[expected_ld], out, smem)
        for i in range(n_elements):
            B[i] = out[i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert expected_st in src
    assert expected_ld in src

    dev = tvm_tirx.cuda(0)
    np_dtype = tvm_tirx.testing.np_dtype_from_str(dtype)
    out = tvm_tirx.runtime.tensor(np.zeros((n_elements,), dtype=np_dtype), dev)
    ex(out)
    np.testing.assert_equal(out.numpy(), np.arange(1, n_elements + 1, dtype=np_dtype))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_forced_vec_dynamic_swizzled_shared_uses_vector_ptx():
    from tvm_tirx.tirx.layout import ComposeLayout

    smem_layout = ComposeLayout(2, 3, 3, TileLayout(S[(64, 8, 32) : (32, 2048, 1)]))

    @T.function
    def kernel(B: T.Tensor((128, 4), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        tid = T.cuda.thread_idx("x")
        smem = T.alloc_tensor(
            ty_args=[T.Tensor((64, 256), "float32", scope="shared", layout=smem_layout)]
        )
        reg = T.alloc_local((4,), "float32", layout="default")
        out = T.alloc_local((4,), "float32", layout="default")
        for i in range(4):
            reg[i] = T.cast(tid * 16 + i + 1, "float32")
        row: T.let = tid % 64
        col: T.let = (tid // 64) * 4
        smem_view = smem.rearrange("r (b c) -> r b c", b=8, c=32)
        T.cuda.tile.apply(
            T.ptx["st.shared.v4.u32"],
            smem_view.sub[row, 0, col : col + 4].view("uint32"),
            reg.view("uint32"),
        )
        T.cuda.cta_sync()
        T.cuda.tile.apply(
            T.ptx["ld.shared.v4.u32"],
            out.view("uint32"),
            smem_view.sub[row, 0, col : col + 4].view("uint32"),
        )
        for i in range(4):
            B[tid, i] = out[i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "copy/fallback" not in src
    assert "st.shared.v4.u32" in src
    assert "ld.shared.v4.u32" in src

    dev = tvm_tirx.cuda(0)
    out = tvm_tirx.runtime.tensor(np.zeros((128, 4), dtype="float32"), dev)
    ex(out)
    expected = np.array([[tid * 16 + i + 1 for i in range(4)] for tid in range(128)], "float32")
    np.testing.assert_equal(out.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_explicit_vector_instructions():
    @T.function
    def kernel(B: T.Tensor((4,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", layout="default")])
        reg = T.alloc_tensor(
            ty_args=[T.Tensor((4,), "float32", scope="local", layout=TileLayout(S[4]))]
        )
        out = T.alloc_tensor(
            ty_args=[T.Tensor((4,), "float32", scope="local", layout=TileLayout(S[4]))]
        )
        for i in range(4):
            reg[i] = T.cast(i + 1, "float32")
        T.cuda.tile.apply(T.ptx["st.shared.v4.u32"], smem.view("uint32"), reg.view("uint32"))
        T.cuda.cta_sync()
        T.cuda.tile.apply(T.ptx["ld.shared.v4.u32"], out.view("uint32"), smem.view("uint32"))
        for i in range(4):
            B[i] = out[i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "tvm_builtin_copy_" not in src
    assert "st.shared.v4.u32" in src
    assert "ld.shared.v4.u32" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_forced_vec_rejects_size_mismatch():
    @T.function
    def kernel(B: T.Tensor((4,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        smem = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", layout="default")])
        reg = T.alloc_local((4,), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["st.shared.v2.b32"], smem, reg)
        B[0] = T.cast(0, "float32")

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="expected trailing atom shape"):
            tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_copy_vector_rejects_missing_thread_axis():
    @T.function
    def kernel(B: T.Tensor((4,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        _lane = T.cuda.lane_id()
        smem = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", layout="default")])
        reg = T.alloc_tensor(
            ty_args=[T.Tensor((4,), "float32", scope="local", layout=TileLayout(S[4]))]
        )
        T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], smem, reg, scope="warp")
        B[0] = T.cast(0, "float32")

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        with pytest.raises(RuntimeError, match="expected leading group shape"):
            tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")


@pytest.mark.gpu
@pytest.mark.parametrize("words", [2, 4])
def test_reg_explicit_vector_width(words):
    # Same eight fp16 elements as the old vector-selection unit test. Both
    # instruction widths are legal; the caller chooses one explicitly.
    opcode = f"st.shared.v{words}.b32"

    @T.function
    def kernel(B: T.Tensor((8,), "float16")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        reg = T.alloc_local((8,), "float16", layout=TileLayout(S[8:1]))
        smem = T.alloc_shared((8,), "float16", layout=TileLayout(S[8:1]))
        for i in range(8):
            reg[i] = T.cast(i + 1, "float16")
        T.cuda.tile.apply(
            T.ptx[opcode],
            smem.view("uint32").view(4 // words, words),
            reg.view("uint32").view(4 // words, words),
        )
        for i in range(8):
            B[i] = smem[i]

    target = tvm_tirx.target.Target("cuda")
    with target:
        ex = tvm_tirx.tirx.compile(kernel, target=target)
    assert _native_call_count(ex, opcode) == 4 // words
    out = tvm_tirx.runtime.tensor(np.zeros(8, "float16"), tvm_tirx.cuda())
    ex(out)
    np.testing.assert_array_equal(out.numpy(), np.arange(1, 9, dtype="float16"))


@pytest.mark.parametrize("case", ["wg", "wg_slice", "tcgen05"])
def test_reg_view_preserves_shared_and_register_addresses(case):
    from tvm_tirx.tirx.layout import ComposeLayout

    if case == "tcgen05":
        m, n, r_layout, s_layout = _tcgen05_d_epilogue_layouts()
        shape, dtype, start, width = (m, n), "float32", 0, n
        recipe = "(w h r) (k l v) -> w (r l) k h v"
        factors = dict(w=4, h=2, r=8, k=8, l=4, v=2)
    else:
        shape, dtype = (128, 64), "float16"
        start, width = (8, 32) if case == "wg_slice" else (0, 64)
        r_layout = TileLayout(S[(128, 64) : (1 @ CUDAAxis.tid_in_wg, 1)])
        s_layout = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
        recipe = "(w t) (r v) -> w t r v"
        factors = dict(t=32, v=8)

    @T.function
    def views() -> None:
        T.device_entry()
        reg = T.alloc_local(shape, dtype, layout=r_layout)
        smem = T.alloc_shared(shape, dtype, layout=s_layout)
        r_view = reg.sub[:, start : start + width].rearrange(recipe, **factors)
        s_view = smem.sub[:, start : start + width].rearrange(recipe, **factors)
        T.evaluate(r_view.data)
        T.evaluate(s_view.data)

    tensors = []

    def visit(node):
        if isinstance(node, tvm_tirx.ir.Bind) and isinstance(node.var.ty, tvm_tirx.tirx.TensorType):
            tensors.append(node.var)

    tvm_ffi.structural_walk(views.body, visit)
    r_view = next(t for t in reversed(tensors) if t.ty.scope == MemoryScope("local"))
    s_view = next(t for t in reversed(tensors) if t.ty.scope == MemoryScope("shared"))
    analyzer = tvm_tirx.sym.Analyzer()
    # Check all scalar coordinates, including the fp16 elements inside a word.
    for tid in range(128):
        for value in range(width if case != "tcgen05" else 32):
            if case == "tcgen05":
                row = tid // 32 * 16 + tid % 32 // 4 + (value // 2 % 2) * 8
                col = value % 2 + (tid % 4) * 2 + (value // 4) * 8
                coords = (tid // 32, tid % 32, value // 4, value // 2 % 2, value % 2)
            else:
                row, col = tid, start + value
                coords = (tid // 32, tid % 32, value // 8, value % 8)
            for layout, view in ((r_layout, r_view), (s_layout, s_view)):
                before = layout.apply(row, col, shape=shape)
                after = dict(view.ty.layout.apply(*coords, shape=view.ty.shape))
                after["m"] = after.get("m", 0) + view.ty.elem_offset
                assert {k: int(analyzer.simplify(v)) for k, v in before.items()} == {
                    k: int(analyzer.simplify(v)) for k, v in after.items()
                }


# --- tcgen05 D epilogue deposit (tf32_hc_prenorm_gemm) -----------------------
# Register-to-shared epilogue deposit after ``tcgen05.ld``
# pulls the M=64 accumulator fragment from TMEM into ``d_reg``, then deposits it
# into 128B-swizzled MMA SMEM for the subsequent TMA store to gmem D.
_TCGEN05_D_ATOM = "16x256b"
_TCGEN05_D_SHAPE = (64, 64)
_TCGEN05_D_DTYPE = "float32"
_TCGEN05_D_SWIZZLE = 3  # SwizzleMode 128B → mma_shared_layout(..., 3, shape)
_TCGEN05_D_SLICE = (slice(0, 64), slice(0, 64))


def _tcgen05_d_epilogue_layouts():
    from tvm_tirx.backend.cuda.layout import mma_shared_layout

    m, n = _TCGEN05_D_SHAPE
    reg_layout = tcgen05_register_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    return m, n, reg_layout, smem_layout


def _build_tcgen05_d_epilogue_deposit():
    """Explicit 64-bit stores: R (tcgen05 atom) → S (128B swizzle)."""
    from tvm_tirx.backend.cuda.layout import mma_shared_layout

    m, n = _TCGEN05_D_SHAPE
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    reg_layout = tcgen05_register_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    sl_m, sl_n = _TCGEN05_D_SLICE

    @T.function
    def deposit(
        d_reg: T.Tensor((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout),
    ) -> None:
        smem_cd_mma = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), _TCGEN05_D_DTYPE, scope="shared", layout=smem_layout)]
        )
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _lane = T.cuda.lane_id()
        T.cuda.tile.apply(
            T.ptx["st.shared.v2.u32"],
            _epilogue_pairs(smem_cd_mma.sub[sl_m, sl_n]),
            _epilogue_pairs(d_reg.sub[sl_m, sl_n]),
            scope="warpgroup",
        )

    return deposit


def _compile_tcgen05_d_epilogue_deposit():
    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": _build_tcgen05_d_epilogue_deposit()})
        return tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")


def test_reg_copy_tcgen05_d_epilogue_deposit_layout_pairing():
    # The former solver regression lost half the stores by fusing different
    # register groups. Verify every original physical address after the view.
    test_reg_view_preserves_shared_and_register_addresses("tcgen05")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_reg_copy_tcgen05_d_epilogue_deposit_codegen():
    # Each lane owns 32 fp32 values. Its sixteen 64-bit stores must all survive.
    ex = _compile_tcgen05_d_epilogue_deposit()
    assert _native_call_count(ex, "st.shared.v2.u32") == 16


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
    """Round-trip ``d_reg`` through shared memory and dump each thread's registers."""
    from tvm_tirx.backend.cuda.layout import mma_shared_layout

    m, n = _TCGEN05_D_SHAPE
    regs_per_thread = 32  # ``.16x256b.x8`` fp32: 4 regs/slot x rep=8
    smem_layout = mma_shared_layout(_TCGEN05_D_DTYPE, _TCGEN05_D_SWIZZLE, (m, n))
    reg_layout = tcgen05_register_layout(_TCGEN05_D_ATOM, (m, n), _TCGEN05_D_DTYPE)
    sl_m, sl_n = _TCGEN05_D_SLICE

    @T.function
    def kernel(
        A: T.Tensor((m, n), _TCGEN05_D_DTYPE), B: T.Tensor((m, n), _TCGEN05_D_DTYPE)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _lane = T.cuda.lane_id()
        tid_wg = T.cuda.thread_in_warpgroup()
        lane = T.cuda.lane_id()
        d_reg = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout)]
        )
        d_reg_out = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), _TCGEN05_D_DTYPE, scope="local", layout=reg_layout)]
        )
        smem_cd_mma = T.alloc_tensor(
            ty_args=[T.Tensor((m, n), _TCGEN05_D_DTYPE, scope="shared", layout=smem_layout)]
        )
        reg_in = d_reg.view(regs_per_thread, layout="default")
        reg_out = d_reg_out.view(regs_per_thread, layout="default")
        for r in T.serial(regs_per_thread):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            reg_in[r] = A[row, col]
        T.cuda.tile.apply(
            T.ptx["st.shared.v2.u32"],
            _epilogue_pairs(smem_cd_mma.sub[sl_m, sl_n]),
            _epilogue_pairs(d_reg.sub[sl_m, sl_n]),
            scope="warpgroup",
        )
        T.cuda.cta_sync()
        T.cuda.tile.apply(
            T.ptx["ld.shared.v2.u32"],
            _epilogue_pairs(d_reg_out.sub[sl_m, sl_n]),
            _epilogue_pairs(smem_cd_mma.sub[sl_m, sl_n]),
            scope="warpgroup",
        )
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
    explicit 64-bit shared stores then the corresponding loads, and dumps ``d_reg_out`` back to gmem
    ``B`` through explicit physical-register views. Pre-fix pairing dropped 2/3 of the S groups —
    ``max|B-A|`` was hundreds, not 0.
    """
    m, n = _TCGEN05_D_SHAPE
    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": _build_tcgen05_d_epilogue_deposit_roundtrip()}),
            target=target,
            pipeline="tirx",
        )

    rows = np.arange(m, dtype=np.int32)[:, None]
    cols = np.arange(n, dtype=np.int32)[None, :]
    a_np = (rows * 100 + cols).astype(np.float32)
    b_np = np.zeros((m, n), dtype=np.float32)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(b_np, dev)
        mod(a, b)
        np.testing.assert_allclose(b.numpy(), a_np, rtol=0, atol=0)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
