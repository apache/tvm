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

"""Native cvt instructions over explicit register views."""

import math

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tirx.layout_fixtures import tcgen05_register_layout
from tvm.backend.cuda.axis import Axis as CUDAAxis
from tvm.backend.cuda.axis import laneid, tid_in_wg, tx, warpid
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


def _run(kernel, arrays, expected, output=0, atol=1e-3, rtol=1e-7, target="cuda"):
    with tvm.target.Target(target):
        module = tvm.tirx.compile(kernel, target=target, pipeline="tirx")

    def check():
        args = [tvm.runtime.tensor(a, tvm.cuda()) for a in arrays]
        module(*args)
        tvm.testing.assert_allclose(args[output].numpy(), expected, atol=atol, rtol=rtol)

    tvm.testing.run_with_gpu_lock(check)
    return module


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("shape", [(8,), (16, 16), (5, 5)])
@pytest.mark.parametrize("A_dtype", ["float16", "float32"])
@pytest.mark.parametrize("B_dtype", ["float16", "float32"])
def test_cast_thread_local(shape, A_dtype, B_dtype):
    A_ref = np.random.rand(*shape).astype(A_dtype)
    B_ref = np.random.rand(*shape).astype(B_dtype)
    B_ref = A_ref.astype(B_dtype)

    words = {"float16": "uint16", "float32": "float32"}
    load = {"float16": "ld.global.b16", "float32": "ld.global.f32"}[A_dtype]
    store = {"float16": "st.global.b16", "float32": "st.global.f32"}[B_dtype]
    opcode = {
        ("float16", "float32"): "cvt.f32.f16",
        ("float32", "float16"): "cvt.rn.f16.f32",
        ("float16", "float16"): "mov.b16",
        ("float32", "float32"): "mov.f32",
    }[A_dtype, B_dtype]
    count = math.prod(shape)

    @T.function
    def test_cast(
        A: T.Tensor(shape, A_dtype, layout=TileLayout(S[shape])),
        B: T.Tensor(shape, B_dtype, layout=TileLayout(S[shape])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(256,)))
        cta_id = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        A_local = T.alloc_local(shape, dtype=A_dtype, layout=TileLayout(S[shape]))
        B_local = T.alloc_local(shape, dtype=B_dtype, layout=TileLayout(S[shape]))
        # One thread owns this entire tensor; other launched threads stay idle.
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx[load],
                A_local.view(words[A_dtype]).view(count, 1),
                A.view(words[A_dtype]).view(count, 1),
            )
            T.cuda.tile.apply(
                T.ptx[opcode],
                B_local.view(words[B_dtype]).view(count, 1),
                A_local.view(words[A_dtype]).view(count, 1),
            )
            T.cuda.tile.apply(
                T.ptx[store],
                B.view(words[B_dtype]).view(count, 1),
                B_local.view(words[B_dtype]).view(count, 1),
            )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_warpgroup_local_view(A_dtype, B_dtype):
    """T.cast in warpgroup scope with offset (tid_in_wg + layout offset). Covers offset/tid_in_wg/warpgroup scope."""  # noqa: E501
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    use_offset = True
    if use_offset:
        from tvm.tirx.layout import Axis, Iter

        m_axis = Axis.get("m")
        shard = [Iter(N_THREADS, 1, tid_in_wg), Iter(LOCAL_LEN, 1, m_axis)]
        cast_layout = TileLayout.from_iters(shard, [], {m_axis: 0})
    else:
        cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tid_in_wg, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tid_in_wg, i]
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.view("uint16").view(4, 32, LOCAL_LEN, 1),
            reg_src_view.view(4, 32, LOCAL_LEN, 1),
            scope="warpgroup",
        )
        for i in T.serial(LOCAL_LEN):
            B[tid_in_wg, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_cta_local_view(A_dtype, B_dtype):
    """T.cast with view+layout in CTA scope (128 threads, register->register)."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N_THREADS,)))
        cta_id = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tx_var, i]
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.view("uint16").view(4, 32, LOCAL_LEN, 1),
            reg_src_view.view(4, 32, LOCAL_LEN, 1),
            scope="cta",
        )
        for i in T.serial(LOCAL_LEN):
            B[tx_var, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
def test_cast_warpgroup_src_layout_to_flat_uses_vec2_intrinsic(A_dtype, B_dtype):
    """Regression: GEMM-epilogue cast pattern must emit the packed vec2 cuda intrinsic.

    Pattern: both sides have a per-thread 1xK row layout. dst is
    allocated per-chunk to keep both operands wg-distributed — the dispatch
    requires layout-symmetric operands (no flat-vs-wg asymmetry).
    """

    N_THREADS, LOCAL_LEN, N_CHUNKS = 128, 8, 4
    g_shape = (N_THREADS, LOCAL_LEN * N_CHUNKS)
    g_layout = TileLayout(S[g_shape])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref = A_ref.astype(B_dtype)

    opcode = "cvt.rn.f16x2.f32" if B_dtype == "float16" else "cvt.rn.bf16x2.f32"

    @T.function
    def test_cast(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        for no in T.unroll(N_CHUNKS):
            reg_src = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (LOCAL_LEN,),
                        A_dtype,
                        scope="local",
                        layout="default",
                    )
                ],
            )
            Dreg_chunk = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (LOCAL_LEN,),
                        B_dtype,
                        scope="local",
                        layout="default",
                    )
                ],
            )
            for i in T.serial(LOCAL_LEN):
                reg_src[i] = A[tid, no * LOCAL_LEN + i]
            reg_src_view = reg_src.view(
                N_THREADS,
                LOCAL_LEN,
                layout=TileLayout(S[(128, LOCAL_LEN) : (1 @ CUDAAxis.tid_in_wg, 1)]),
            )
            Dreg_chunk_view = Dreg_chunk.view(
                N_THREADS,
                LOCAL_LEN,
                layout=TileLayout(S[(128, LOCAL_LEN) : (1 @ CUDAAxis.tid_in_wg, 1)]),
            )
            pairs = reg_src_view.view(4, 32, LOCAL_LEN // 2, 2)
            # PTX writes its first f32 input into the high halfword.
            T.cuda.tile.apply(
                T.ptx[opcode],
                Dreg_chunk_view.view("uint32").view(4, 32, LOCAL_LEN // 2, 1),
                pairs[:, :, :, 1:2],
                pairs[:, :, :, 0:1],
                scope="warpgroup",
            )
            for i in T.serial(LOCAL_LEN):
                B[tid, no * LOCAL_LEN + i] = Dreg_chunk[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_cast})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        assert opcode in src, src

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_ref, dev)
            B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
            mod(A, B)
            tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("A_dtype,B_dtype", [("float32", "float16"), ("float32", "bfloat16")])
@pytest.mark.parametrize("slice_start,slice_end", [(0, 4), (2, 6), (4, 8)])
def test_cast_local_view_sliced(A_dtype, B_dtype, slice_start, slice_end):
    """An explicit CTA conversion writes only the selected local slice."""
    N_THREADS, LOCAL_LEN = 128, 8
    g_shape = (N_THREADS, LOCAL_LEN)
    g_layout = TileLayout(S[g_shape])
    cast_layout = TileLayout(S[(N_THREADS, LOCAL_LEN) : (1 @ tx, 1)])

    A_ref = np.random.rand(*g_shape).astype(A_dtype)
    B_ref = np.zeros(g_shape, dtype=B_dtype)
    B_ref[:, slice_start:slice_end] = A_ref[:, slice_start:slice_end].astype(B_dtype)

    opcode = "cvt.rn.f16.f32" if B_dtype == "float16" else "cvt.rn.bf16.f32"

    @T.function
    def kernel(
        A: T.Tensor(g_shape, A_dtype, layout=g_layout),
        B: T.Tensor(g_shape, B_dtype, layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N_THREADS,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        reg_src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    A_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        reg_dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (LOCAL_LEN,),
                    B_dtype,
                    scope="local",
                    layout="default",
                )
            ],
        )
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[tx, i]
            reg_dst[i] = T.cast(0, B_dtype)
        reg_src_view = reg_src.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        reg_dst_view = reg_dst.view(N_THREADS, LOCAL_LEN, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx[opcode],
            reg_dst_view.sub[:, slice_start:slice_end]
            .view("uint16")
            .view(4, 32, slice_end - slice_start, 1),
            reg_src_view.sub[:, slice_start:slice_end].view(4, 32, slice_end - slice_start, 1),
            scope="cta",
        )
        for i in T.serial(LOCAL_LEN):
            B[tx, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_ref, dev)
        B = tvm.runtime.tensor(np.zeros(g_shape, dtype=B_dtype), dev)
        mod(A, B)
        tvm.testing.assert_allclose(B.numpy(), B_ref, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("slice_start,slice_end", [(0, 2), (2, 4)])
def test_cast_mixed_axes_and_subregion(slice_start, slice_end):
    """Test cast with mixed axes and subregion."""

    N_WARPS, LANES = 2, 32
    LOCAL_LEN = 4
    full_shape = (8, N_WARPS, 4, LOCAL_LEN)
    g_layout = TileLayout(S[full_shape])
    cast_layout = TileLayout(S[full_shape : (4 @ laneid, 1 @ warpid, 1 @ laneid, 1)])

    A_ref = np.zeros(full_shape, dtype="float32")
    for j in range(full_shape[0]):
        for w in range(full_shape[1]):
            for k in range(full_shape[2]):
                for i in range(full_shape[3]):
                    A_ref[j, w, k, i] = float(j * 1000 + w * 100 + k * 10 + i)
    B_ref = np.zeros(full_shape, dtype="float16")
    B_ref[:, :, :, slice_start:slice_end] = A_ref[:, :, :, slice_start:slice_end].astype("float16")

    @T.function
    def kernel(
        A: T.Tensor(full_shape, "float32", layout=g_layout),
        B: T.Tensor(full_shape, "float16", layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=N_WARPS * 32))
        cta_id = T.cuda.block_idx("x")
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        reg_src = T.alloc_tensor(
            ty_args=[T.Tensor((LOCAL_LEN,), "float32", scope="local", layout="default")]
        )
        reg_dst = T.alloc_tensor(
            ty_args=[T.Tensor((LOCAL_LEN,), "float16", scope="local", layout="default")]
        )
        j, k = lane_id // 4, lane_id % 4
        for i in T.serial(LOCAL_LEN):
            reg_src[i] = A[j, warp_id, k, i]
            reg_dst[i] = T.float16(0)
        reg_src_view = reg_src.view(*full_shape, layout=cast_layout)
        reg_dst_view = reg_dst.view(*full_shape, layout=cast_layout)
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            reg_dst_view.sub[:, :, :, slice_start:slice_end]
            .permute(1, 0, 2, 3)
            .view("uint16")
            .view(2, 32, slice_end - slice_start, 1),
            reg_src_view.sub[:, :, :, slice_start:slice_end]
            .permute(1, 0, 2, 3)
            .view(2, 32, slice_end - slice_start, 1),
            scope="cta",
        )
        j_1, k_1 = lane_id // 4, lane_id % 4
        for i in T.serial(LOCAL_LEN):
            B[j_1, warp_id, k_1, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_ref, dev)
        B = tvm.runtime.tensor(np.zeros(full_shape, dtype="float16"), dev)
        mod(A, B)
        tvm.testing.assert_allclose(
            B.numpy(),
            B_ref,
            atol=1e-2,
            rtol=0,
        )

    tvm.testing.run_with_gpu_lock(run_and_check)


def test_cast_validate_extent_mismatch_rejected():
    """Validation rejects when src and dst layouts have same thread positions but different extents."""  # noqa: E501

    view_shape = (2, 8, 4, 8)
    g_layout = TileLayout(S[view_shape])
    src_layout = TileLayout(S[view_shape : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)])
    dst_layout = TileLayout(
        S[view_shape : (2 @ warpid, 8 @ laneid, 1 @ laneid, 1)]
    )  # dim1 extent 8 != 4

    @T.function
    def kernel(
        A: T.Tensor(view_shape, "float32", layout=g_layout),
        B: T.Tensor(view_shape, "float16", layout=g_layout),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 32))
        cta_id = T.cuda.block_idx("x")
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        reg_src = T.alloc_tensor(
            ty_args=[T.Tensor((8,), "float32", scope="local", layout="default")]
        )
        reg_dst = T.alloc_tensor(
            ty_args=[T.Tensor((8,), "float16", scope="local", layout="default")]
        )
        j, k = lane_id // 4, lane_id % 4
        for i in T.serial(8):
            reg_src[i] = A[warp_id, j, k, i]
        reg_src_view = reg_src.view(*view_shape, layout=src_layout)
        reg_dst_view = reg_dst.view(*view_shape, layout=dst_layout)
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            reg_dst_view.view("uint16").view(2, 32, 8, 1),
            reg_src_view.view(2, 32, 8, 1),
            scope="cta",
        )
        j_1, k_1 = lane_id // 4, lane_id % 4
        for i in T.serial(8):
            B[warp_id, j_1, k_1, i] = reg_dst[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        # The mismatched dst also fails the scope-level check (thread axes don't
        # span the full CTA), which fires first — either rejection is fine.
        with pytest.raises(
            Exception,
            match="incorrect .* owner",
        ):
            tvm.tirx.compile(mod, target=target, pipeline="tirx")


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,intrinsic",
    [
        ("float32", "float16", "__float22half2_rn"),
        ("float16", "float32", "__half22float2"),
    ],
)
def test_cast_vec2_packed_dispatch(src_dtype, dst_dtype, intrinsic):
    """Packed narrowing and scalar widening preserve the original pair values."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    load = "ld.global.f32" if src_dtype == "float32" else "ld.global.b16"
    store = "st.global.f32" if dst_dtype == "float32" else "st.global.b16"
    src_word = "float32" if src_dtype == "float32" else "uint16"
    dst_word = "float32" if dst_dtype == "float32" else "uint16"

    @T.function
    def k(
        A: T.Tensor(shape, src_dtype, layout=lay), B: T.Tensor(shape, dst_dtype, layout=lay)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ra = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], src_dtype, scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rb = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], dst_dtype, scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        T.cuda.tile.apply(
            T.ptx[load], ra.view(src_word).view(32, 1), A.sub[tx, :].view(src_word).view(32, 1)
        )
        if T.constexpr(src_dtype == "float32"):
            pairs = ra.view(16, 2)
            T.cuda.tile.apply(
                T.ptx["cvt.rn.f16x2.f32"],
                rb.view("uint32").view(16, 1),
                pairs[:, 1:2],
                pairs[:, 0:1],
            )
        else:
            # __half22float2 expands to two scalar PTX conversions; there is
            # no native cvt.f32x2.f16x2 instruction.
            T.cuda.tile.apply(T.ptx["cvt.f32.f16"], rb.view(32, 1), ra.view("uint16").view(32, 1))
        T.cuda.tile.apply(
            T.ptx[store], B.sub[tx, :].view(dst_word).view(32, 1), rb.view(dst_word).view(32, 1)
        )

    target = tvm.target.Target(
        {"kind": "cuda", "arch": "sm_80", "backend_config": {"cuda": {"target_format": "ptx"}}}
    )
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    opcode = "cvt.rn.f16x2.f32" if src_dtype == "float32" else "cvt.f32.f16"
    assert opcode in src, src
    rng = np.random.RandomState(0)
    a = rng.rand(*shape).astype(src_dtype)
    _run(
        k,
        [a, np.zeros(shape, dtype=dst_dtype)],
        a.astype(dst_dtype),
        output=1,
        target=target,
        atol=0,
        rtol=0,
    )


_SL_ROWS, _SL_COLS = 128, 8


def _sl_compile(fn):
    target = tvm.target.Target("cuda")
    with target:
        tvm.tirx.compile(tvm.IRModule({"main": fn}), target=target, pipeline="tirx")


def test_cast_wg_rejects_thread_local_view():
    """warpgroup conversion on a physical-register (thread-axis-stripped) view is rejected."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view(_SL_COLS, layout="default").view("uint16").view(8, 1),
            src.view(_SL_COLS, layout="default").view(8, 1),
            scope="warpgroup",
        )
        dst_row = dst.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_cta_rejects_thread_local_view():
    """CTA conversion on a physical-register view is rejected (cta -> tx)."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(_SL_ROWS,)))
        _bx = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tx, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tx, 1)]),
                )
            ]
        )
        src_row = src.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tx_var, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view(_SL_COLS, layout="default").view("uint16").view(8, 1),
            src.view(_SL_COLS, layout="default").view(8, 1),
            scope="cta",
        )
        dst_row = dst.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            B[tx_var, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_wg_rejects_partial_thread_coverage():
    """A tid_in_wg layout covering only 64 of the 128 wg threads is rejected."""
    half = 64

    @T.function
    def kernel(
        A: T.Tensor((half, _SL_COLS), "float32", layout=TileLayout(S[half, _SL_COLS])),
        B: T.Tensor((half, _SL_COLS), "float16", layout=TileLayout(S[half, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (half, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(half, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (half, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(half, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view("uint16").view(-1, 32, _SL_COLS, 1),
            src.view(-1, 32, _SL_COLS, 1),
            scope="warpgroup",
        )
        dst_row = dst.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    with pytest.raises(Exception, match="leading group shape"):
        _sl_compile(kernel)


def test_cast_wg_accepts_wg_level_layout():
    """warpgroup conversion on a wg-level (tid_in_wg-distributed) layout compiles."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        _wg = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float32",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (_SL_ROWS, _SL_COLS),
                    "float16",
                    scope="local",
                    layout=TileLayout(S[(_SL_ROWS, _SL_COLS) : (1 @ tid_in_wg, 1)]),
                )
            ]
        )
        src_row = src.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            src_row[i] = A[tid, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"],
            dst.view("uint16").view(-1, 32, _SL_COLS, 1),
            src.view(-1, 32, _SL_COLS, 1),
            scope="warpgroup",
        )
        dst_row = dst.view(_SL_COLS, layout="default")
        for i in T.serial(_SL_COLS):
            B[tid, i] = dst_row[i]

    _sl_compile(kernel)


def test_cast_thread_accepts_local_view():
    """thread scope is exempt: a thread-axis-free local tile still compiles."""

    @T.function
    def kernel(
        A: T.Tensor((_SL_ROWS, _SL_COLS), "float32", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
        B: T.Tensor((_SL_ROWS, _SL_COLS), "float16", layout=TileLayout(S[_SL_ROWS, _SL_COLS])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(_SL_ROWS,)))
        _bx = T.cuda.block_idx("x")
        tx_var = T.cuda.thread_idx("x")
        src = T.alloc_tensor(
            ty_args=[
                T.Tensor((_SL_COLS,), "float32", scope="local", layout=TileLayout(S[(_SL_COLS,)]))
            ]
        )
        dst = T.alloc_tensor(
            ty_args=[
                T.Tensor((_SL_COLS,), "float16", scope="local", layout=TileLayout(S[(_SL_COLS,)]))
            ]
        )
        for i in T.serial(_SL_COLS):
            src[i] = A[tx_var, i]
        T.cuda.tile.apply(
            T.ptx["cvt.rn.f16.f32"], dst.view("uint16").view(_SL_COLS, 1), src.view(_SL_COLS, 1)
        )
        for i in T.serial(_SL_COLS):
            B[tx_var, i] = dst[i]

    _sl_compile(kernel)


_TCGEN05_M, _TCGEN05_K = 64, 64


_TCGEN05_REGS_PER_THREAD = 32  # ``.16x256b`` fp32 atom on (64, 64)


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


def _tcgen05_cast_atom_layout():
    # Production tf32_hc_prenorm_gemm pattern: bf16 payload in fp32 atom slots.
    return tcgen05_register_layout("16x256b", (_TCGEN05_M, _TCGEN05_K), "float32")


def _tcgen05_cast_warpgroup_kernel():
    """Scatter gmem bf16 → reg, ``warpgroup conversion``, gather fp32 → gmem."""
    atom_layout = _tcgen05_cast_atom_layout()
    m, k = _TCGEN05_M, _TCGEN05_K

    @T.function
    def kernel(A: T.Tensor((m, k), "bfloat16"), B: T.Tensor((m, k), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _lane = T.cuda.lane_id()
        tid_wg = T.cuda.thread_in_warpgroup()
        lane = T.cuda.lane_id()
        a_bf16 = T.alloc_tensor(
            ty_args=[T.Tensor((m, k), "bfloat16", scope="local", layout=atom_layout)]
        )
        a_fp32 = T.alloc_tensor(
            ty_args=[T.Tensor((m, k), "float32", scope="local", layout=atom_layout)]
        )
        reg_in = a_bf16.view(_TCGEN05_REGS_PER_THREAD, layout="default")
        reg_out = a_fp32.view(_TCGEN05_REGS_PER_THREAD, layout="default")
        for r in T.serial(_TCGEN05_REGS_PER_THREAD):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            reg_in[r] = A[row, col]
        dst_view = a_fp32.rearrange(
            "(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2
        ).view(4, 32, 8, 2, 2, 1)
        src_view = (
            a_bf16.rearrange("(w h r) (k l v) -> w (r l) k h v", w=4, h=2, r=8, k=8, l=4, v=2)
            .view("uint16")
            .view(4, 32, 8, 2, 2, 1)
        )
        T.cuda.tile.apply(T.ptx["cvt.f32.bf16"], dst_view, src_view, scope="warpgroup")
        for r in T.serial(_TCGEN05_REGS_PER_THREAD):
            row, col = _tcgen05_16x256b_row_col(tid_wg, lane, T.cast(r, "int32"))
            B[row, col] = reg_out[r]

    return kernel


def test_cast_tcgen05_atom_warpgroup_reg_dispatch_compiles():
    """Regression: tf32 A-cast ``warpgroup conversion(a_fp32, a_bf16)`` on tcgen05 atom compiles."""
    _sl_compile(_tcgen05_cast_warpgroup_kernel())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cast_tcgen05_atom_warpgroup_gpu():
    """GPU: bf16→fp32 cast on tcgen05 ``.16x256b`` atom must preserve per-slot layout."""
    m, k = _TCGEN05_M, _TCGEN05_K
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.tirx.compile(
            tvm.IRModule({"main": _tcgen05_cast_warpgroup_kernel()}),
            target=target,
            pipeline="tirx",
        )

    np.random.seed(0)
    a_np = (np.random.randn(m, k).astype("float32") * 4.0).astype("bfloat16")
    b_ref = a_np.astype("float32")

    def run_and_check():
        dev = tvm.cuda(0)
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(np.zeros((m, k), dtype="float32"), dev)
        mod(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_ref, rtol=0, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


def _view_address_proof(layout, shape, permutation, expected):
    @T.function
    def views():
        T.device_entry()
        original = T.alloc_local(shape, "float32", layout=layout)
        canonical = original.permute(*permutation)
        T.evaluate(canonical.data)

    tensors = []

    def collect(node):
        if isinstance(node, tvm.ir.Bind) and isinstance(node.var.ty, tvm.tirx.TensorType):
            tensors.append(node.var)

    tvm_ffi.structural_walk(views.body, collect)
    view = tensors[-1]
    analyzer = tvm.sym.Analyzer()
    for coord in np.ndindex(shape):
        before = layout.apply(*coord, shape=shape)
        after = view.ty.layout.apply(*(coord[i] for i in permutation), shape=view.ty.shape)
        reference = {k: v for k, v in expected(*coord).items() if v != 0}
        for physical in (before, after):
            actual = {str(k): int(analyzer.simplify(v)) for k, v in physical.items()}
            assert {k: v for k, v in actual.items() if v != 0} == reference
    # Replica metadata also survives: a view cannot invent/remove a broadcast.
    assert list(layout.replica) == list(view.ty.layout.replica)


def test_cast_layout_partition_and_validation():
    """All seven original layout samples keep their physical map after a view.

    The old helper's layout-only accept/reject booleans are retired. Instruction
    applicability is checked by the scope/ownership rejection tests below.
    """
    from tvm.tirx.layout import Iter

    cases = [
        (TileLayout(S[(128, 8) : (1 @ tx, 1)]), (128, 8), (0, 1)),
        (TileLayout(S[(128, 8) : (1 @ tid_in_wg, 1)]), (128, 8), (0, 1)),
        (
            TileLayout.from_iters([Iter(4, 16, "m"), Iter(8, 2, tx), Iter(2, 1, "m")]),
            (4, 8, 2),
            (1, 0, 2),
        ),
        (
            TileLayout(S[(2, 8, 4, 2) : (2 @ warpid, 4 @ laneid, 1 @ laneid, 1)]),
            (2, 8, 4, 2),
            (0, 1, 2, 3),
        ),
        (TileLayout(S[(64, 8) : (1, 1)]), (64, 8), (0, 1)),
        (TileLayout(S[(8, 8) : (1 @ tx, 1 @ laneid)]), (8, 8), (0, 1)),
        (
            TileLayout.from_iters([Iter(128, 1, tx), Iter(8, 1, "m")], [Iter(2, 1, laneid)]),
            (128, 8),
            (0, 1),
        ),
    ]
    references = [
        lambda thread, value: {"tx": thread, "m": value},
        lambda thread, value: {"tid_in_wg": thread, "m": value},
        lambda outer, thread, inner: {"tx": 2 * thread, "m": 16 * outer + inner},
        lambda warp, group, thread, value: {
            "warpid": 2 * warp,
            "laneid": 4 * group + thread,
            "m": value,
        },
        lambda row, col: {"m": row + col},
        lambda row, col: {"tx": row, "laneid": col},
        lambda thread, value: {"tx": thread, "m": value},
    ]
    for (layout, shape, permutation), expected in zip(cases, references):
        _view_address_proof(layout, shape, permutation, expected)


def test_cast_joint_decomposition_extents_order():
    """The caller's view keeps both gapped ownership axes in logical order."""
    layout = TileLayout(S[(2, 32, 4) : (2 @ warpid, 32 @ laneid, 1)])
    _view_address_proof(
        layout,
        (2, 32, 4),
        (0, 1, 2),
        lambda warp, lane, value: {"warpid": 2 * warp, "laneid": 32 * lane, "m": value},
    )


if __name__ == "__main__":
    tvm.testing.main()
