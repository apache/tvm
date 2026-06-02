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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import R, S, TileLayout, laneid, wg_local_layout


@pytest.mark.parametrize(
    "src_shape, dst_shape, axes, st_src, st_dst, extent_src, extent_dst",
    [
        # reduce last dim (basic)
        ((32, 32), (32,), (-1,), (0, 0), (0,), (32, 32), (32,)),
        # reduce first dim
        ((32, 32), (32,), (0,), (0, 0), (0,), (32, 32), (32,)),
        # reduce last 2 dims (4D → 2D)
        ((8, 16, 2, 22), (8, 16), (-2, -1), (0, 0, 0, 0), (0, 0), (8, 16, 2, 22), (8, 16)),
        # reduce middle dim (3D → 2D)
        ((4, 8, 6), (4, 6), (1,), (0, 0, 0), (0, 0), (4, 8, 6), (4, 6)),
        # small non-power-of-2
        ((32, 7), (32,), (-1,), (0, 0), (0,), (32, 7), (32,)),
        # with offset/slicing
        ((32, 32), (32,), (-1,), (1, 1), (2,), (5, 8), (5,)),
    ],
)
@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_shared(
    src_shape, dst_shape, axes, st_src, st_dst, extent_src, extent_dst, op_type, dtype, accum
):
    dev = tvm.cuda(0)
    ndim_src = len(src_shape)

    thread_cnt = 32
    if np.prod(src_shape) > 1024:
        thread_cnt = 128

    s_shape_src = src_shape
    s_shape_dst = dst_shape
    copy_slice_src = list(slice(None) for _ in range(ndim_src))
    copy_slice_dst = list(slice(None) for _ in range(len(dst_shape)))
    reduce_slice_src = list(slice(st_src[i], st_src[i] + extent_src[i]) for i in range(ndim_src))
    reduce_slice_dst = list(
        slice(st_dst[i], st_dst[i] + extent_dst[i]) for i in range(len(dst_shape))
    )
    g_layout_src = s_layout_src = TileLayout(S[src_shape])
    g_layout_dst = s_layout_dst = TileLayout(S[dst_shape])

    # fmt: off
    @Tx.prim_func
    def test_reduction(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, src_shape, dtype, layout=g_layout_src)
        B = Tx.match_buffer(B_ptr, dst_shape, dtype, layout=g_layout_dst)

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        _tid = Tx.thread_id([thread_cnt])

        with Tx.cta():
            A_smem = Tx.alloc_buffer(s_shape_src, dtype, scope="shared", layout=s_layout_src)
            B_smem = Tx.alloc_buffer(s_shape_dst, dtype, scope="shared", layout=s_layout_dst)

            Tx.copy(A_smem[tuple(copy_slice_src)], A[tuple(copy_slice_src)])
            if accum:
                Tx.copy(B_smem[tuple(copy_slice_dst)], B[tuple(copy_slice_dst)])
            Tx.cuda.cta_sync()
            if op_type == "sum":
                Tx.sum(B_smem[tuple(reduce_slice_dst)], A_smem[tuple(reduce_slice_src)], axes=axes, accum=accum) # noqa: E501
            elif op_type == "max":
                Tx.max(B_smem[tuple(reduce_slice_dst)], A_smem[tuple(reduce_slice_src)], axes=axes, accum=accum) # noqa: E501
            elif op_type == "min":
                Tx.min(B_smem[tuple(reduce_slice_dst)], A_smem[tuple(reduce_slice_src)], axes=axes, accum=accum) # noqa: E501
            Tx.cuda.cta_sync()
            Tx.copy(B[tuple(copy_slice_dst)], B_smem[tuple(copy_slice_dst)])
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_reduction})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*src_shape).astype(dtype)
        if accum:
            B_np = np.random.rand(*dst_shape).astype(dtype) * 0.5
        else:
            B_np = np.zeros(dst_shape, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np.copy(), dev)
        mod(A, B)

        A_slice = A_np[tuple(reduce_slice_src)]
        if op_type == "sum":
            ref = A_slice.sum(axis=axes)
        elif op_type == "max":
            ref = A_slice.max(axis=axes)
        elif op_type == "min":
            ref = A_slice.min(axis=axes)
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")

        B_old_slice = B_np[tuple(reduce_slice_dst)]
        if accum:
            if op_type == "sum":
                ref = ref + B_old_slice
            elif op_type == "max":
                ref = np.maximum(ref, B_old_slice)
            elif op_type == "min":
                ref = np.minimum(ref, B_old_slice)

        atol = 1e-5 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(ref, B.numpy()[tuple(reduce_slice_dst)], atol=atol)


@pytest.mark.parametrize("exec_scope", ["warp", "warpgroup", "thread"])
@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_shared_subscope(exec_scope, op_type, accum):
    """Test shared reduction at warp/warpgroup/thread exec scope."""
    dev = tvm.cuda(0)
    dtype = "float32"
    src_shape = (4, 8)
    dst_shape = (4,)
    axes = (-1,)

    g_layout_src = s_layout_src = TileLayout(S[src_shape])
    g_layout_dst = s_layout_dst = TileLayout(S[dst_shape])

    # fmt: off
    if exec_scope == "warp":
        @Tx.prim_func
        def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, src_shape, dtype, layout=g_layout_src)
            B = Tx.match_buffer(B_ptr, dst_shape, dtype, layout=g_layout_dst)
            Tx.device_entry()
            warp_id = Tx.warp_id([(256) // 32])
            _bx = Tx.cta_id([1])
            _tid = Tx.thread_id([256])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(list(src_shape), dtype, scope="shared", layout=s_layout_src) # noqa: E501
                B_smem = Tx.alloc_buffer(list(dst_shape), dtype, scope="shared", layout=s_layout_dst) # noqa: E501
                Tx.copy(A_smem, A)
                if accum:
                    Tx.copy(B_smem, B)
                Tx.cuda.cta_sync()
                if warp_id == 5:
                    with Tx.warp():
                        if op_type == "sum":
                            Tx.sum(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "max":
                            Tx.max(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "min":
                            Tx.min(B_smem, A_smem, axes=axes, accum=accum)
                Tx.cuda.cta_sync()
                Tx.copy(B, B_smem)
    elif exec_scope == "warpgroup":
        @Tx.prim_func
        def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, src_shape, dtype, layout=g_layout_src)
            B = Tx.match_buffer(B_ptr, dst_shape, dtype, layout=g_layout_dst)
            Tx.device_entry()
            wg_id = Tx.warpgroup_id([(256) // 128])
            _bx = Tx.cta_id([1])
            _tid = Tx.thread_id([256])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(list(src_shape), dtype, scope="shared", layout=s_layout_src) # noqa: E501
                B_smem = Tx.alloc_buffer(list(dst_shape), dtype, scope="shared", layout=s_layout_dst) # noqa: E501
                Tx.copy(A_smem, A)
                if accum:
                    Tx.copy(B_smem, B)
                Tx.cuda.cta_sync()
                if wg_id == 0:
                    with Tx.warpgroup():
                        if op_type == "sum":
                            Tx.sum(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "max":
                            Tx.max(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "min":
                            Tx.min(B_smem, A_smem, axes=axes, accum=accum)
                Tx.cuda.cta_sync()
                Tx.copy(B, B_smem)
    elif exec_scope == "thread":
        @Tx.prim_func
        def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, src_shape, dtype, layout=g_layout_src)
            B = Tx.match_buffer(B_ptr, dst_shape, dtype, layout=g_layout_dst)
            Tx.device_entry()
            _bx = Tx.cta_id([1])
            _tid = Tx.thread_id([256])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(list(src_shape), dtype, scope="shared", layout=s_layout_src) # noqa: E501
                B_smem = Tx.alloc_buffer(list(dst_shape), dtype, scope="shared", layout=s_layout_dst) # noqa: E501
                Tx.copy(A_smem, A)
                if accum:
                    Tx.copy(B_smem, B)
                Tx.cuda.cta_sync()
                if _tid == 65:
                    with Tx.thread():
                        if op_type == "sum":
                            Tx.sum(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "max":
                            Tx.max(B_smem, A_smem, axes=axes, accum=accum)
                        elif op_type == "min":
                            Tx.min(B_smem, A_smem, axes=axes, accum=accum)
                Tx.cuda.cta_sync()
                Tx.copy(B, B_smem)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*src_shape).astype(dtype)
        if accum:
            B_np = np.random.rand(*dst_shape).astype(dtype) * 0.5
        else:
            B_np = np.zeros(dst_shape, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np.copy(), dev)
        mod(A, B)

        if op_type == "sum":
            ref = A_np.sum(axis=-1)
            if accum:
                ref = ref + B_np
        elif op_type == "max":
            ref = A_np.max(axis=-1)
            if accum:
                ref = np.maximum(ref, B_np)
        elif op_type == "min":
            ref = A_np.min(axis=-1)
            if accum:
                ref = np.minimum(ref, B_np)

        tvm.testing.assert_allclose(ref, B.numpy(), atol=1e-5)


@pytest.mark.parametrize(
    "src_shape, dst_shape, axes",
    [
        ((1,), (1,), (0,)),
        ((4,), (1,), (0,)),
        ((7,), (1,), (0,)),
        ((16,), (1,), (0,)),
        ((32,), (1,), (0,)),
        ((4, 8), (8,), (0,)),
        ((4, 8), (4,), (1,)),
        ((3, 4, 5), (4,), (0, 2)),
        ((2, 3, 4), (2, 3), (-1,)),
        ((2, 3, 4), (3, 4), (0,)),
    ],
)
@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_local_thread_wise(src_shape, dst_shape, axes, op_type, accum):
    """Test thread-wise local reduction with various shapes and axes."""
    dev = tvm.cuda(0)
    dtype = "float32"
    src_total = 1
    for s in src_shape:
        src_total *= s
    dst_total = 1
    for s in dst_shape:
        dst_total *= s

    def decompose_flat(flat_idx, shape):
        indices = []
        rem = flat_idx
        for s in reversed(list(shape)):
            indices.append(rem % s)
            rem = rem // s
        indices.reverse()
        return indices

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, list(src_shape), dtype, layout=TileLayout(S[src_shape]))
        B = Tx.match_buffer(B_ptr, list(dst_shape), dtype, layout=TileLayout(S[dst_shape]))

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        _tid = Tx.thread_id([1])

        with Tx.thread():
            A_local = Tx.alloc_buffer(list(src_shape), dtype, scope="local")
            B_local = Tx.alloc_buffer(list(dst_shape), dtype, scope="local")

            for i in Tx.serial(src_total):
                idx = Tx.meta_var(decompose_flat(i, src_shape))
                A_local[tuple(idx)] = A[tuple(idx)]

            if accum:
                for i in Tx.serial(dst_total):
                    idx = Tx.meta_var(decompose_flat(i, dst_shape))
                    B_local[tuple(idx)] = B[tuple(idx)]

            if op_type == "sum":
                Tx.sum(B_local, A_local, axes=axes, accum=accum)
            elif op_type == "max":
                Tx.max(B_local, A_local, axes=axes, accum=accum)
            elif op_type == "min":
                Tx.min(B_local, A_local, axes=axes, accum=accum)

            for i in Tx.serial(dst_total):
                idx = Tx.meta_var(decompose_flat(i, dst_shape))
                B[tuple(idx)] = B_local[tuple(idx)]
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*src_shape).astype(dtype)
        if accum:
            B_np = np.random.rand(*dst_shape).astype(dtype) * 0.5
        else:
            B_np = np.zeros(dst_shape, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np.copy(), dev)
        mod(A, B)

        if op_type == "sum":
            ref = A_np.sum(axis=axes)
            if accum:
                ref = ref + B_np
        elif op_type == "max":
            ref = A_np.max(axis=axes)
            if accum:
                ref = np.maximum(ref, B_np)
        elif op_type == "min":
            ref = A_np.min(axis=axes)
            if accum:
                ref = np.minimum(ref, B_np)

        tvm.testing.assert_allclose(ref.reshape(B_np.shape), B.numpy(), atol=1e-5)


@pytest.mark.parametrize(
    "inner_dims, dst_dims, axes, accum, slice_end",
    [
        # 2D: reduce last dim
        ((64,), (1,), (-1,), False, None),
        ((64,), (1,), (-1,), True, None),
        # 2D: sliced reduce
        ((64,), (1,), (-1,), False, 32),
        # 3D: reduce both inner dims
        ((4, 8), (1, 1), (1, 2), False, None),
        # 3D: reduce last dim only
        ((4, 8), (4, 1), (-1,), False, None),
        # 3D: reduce middle dim only
        ((4, 8), (1, 8), (1,), False, None),
    ],
)
@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
def test_reduction_local_view_basic(inner_dims, dst_dims, axes, accum, slice_end, op_type):
    """Test view-based local reduction with simple purely-local layouts."""
    dev = tvm.cuda(0)
    dtype = "float32"
    thread_cnt = 32

    src_shape = (32, *inner_dims)
    dst_shape = (32, *dst_dims)

    def row_major_strides(dims):
        strides = []
        s = 1
        for d in reversed(dims):
            strides.insert(0, s)
            s *= d
        return strides

    acc_view_layout = Tx.TileLayout(
        Tx.S[src_shape : (1 @ laneid, *tuple(row_major_strides(inner_dims)))]
    )
    red_view_layout = Tx.TileLayout(
        Tx.S[dst_shape : (1 @ laneid, *tuple(row_major_strides(dst_dims)))]
    )
    g_layout_a = TileLayout(S[src_shape])
    g_layout_b = TileLayout(S[dst_shape])

    src_local_total = 1
    for d in inner_dims:
        src_local_total *= d
    dst_local_total = 1
    for d in dst_dims:
        dst_local_total *= d

    def decompose_flat(flat_idx, shape):
        indices = []
        rem = flat_idx
        for s in reversed(list(shape)):
            indices.append(rem % s)
            rem = rem // s
        indices.reverse()
        return indices

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, list(src_shape), dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, list(dst_shape), dtype, layout=g_layout_b)

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        _warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([thread_cnt])

        acc = Tx.alloc_buffer(list((1, *inner_dims)), dtype=dtype, scope="local", layout=g_layout_a)
        red = Tx.alloc_buffer(list((1, *dst_dims)), dtype=dtype, scope="local", layout=g_layout_b)

        with Tx.thread():
            for i in Tx.serial(src_local_total):
                idx = Tx.meta_var(decompose_flat(i, inner_dims))
                acc[(0, *list(idx))] = A[(lane_id, *list(idx))]
            if accum:
                for i in Tx.serial(dst_local_total):
                    idx = Tx.meta_var(decompose_flat(i, dst_dims))
                    red[(0, *list(idx))] = B[(lane_id, *list(idx))]
        with Tx.warp():
            acc_view = acc.view(*src_shape, layout=acc_view_layout)
            red_view = red.view(*dst_shape, layout=red_view_layout)
            if slice_end is not None:
                if op_type == "sum":
                    Tx.sum(red_view, acc_view[:, slice_end // 2:slice_end], axes=axes, accum=accum)
                elif op_type == "max":
                    Tx.max(red_view, acc_view[:, slice_end // 2:slice_end], axes=axes, accum=accum)
                elif op_type == "min":
                    Tx.min(red_view, acc_view[:, slice_end // 2:slice_end], axes=axes, accum=accum)
            else:
                if op_type == "sum":
                    Tx.sum(red_view, acc_view, axes=axes, accum=accum)
                elif op_type == "max":
                    Tx.max(red_view, acc_view, axes=axes, accum=accum)
                elif op_type == "min":
                    Tx.min(red_view, acc_view, axes=axes, accum=accum)

        with Tx.thread():
            for i in Tx.serial(dst_local_total):
                idx = Tx.meta_var(decompose_flat(i, dst_dims))
                B[(lane_id, *list(idx))] = red[(0, *list(idx))]
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*src_shape).astype(dtype)
        if accum:
            B_np = np.random.rand(*dst_shape).astype(dtype) * 0.5
        else:
            B_np = np.zeros(dst_shape, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np.copy(), dev)
        mod(A, B)

        A_data = A_np[:, slice_end // 2 : slice_end] if slice_end is not None else A_np
        if op_type == "sum":
            ref = A_data.sum(axis=axes, keepdims=True)
            if accum:
                ref = ref + B_np
        elif op_type == "max":
            ref = A_data.max(axis=axes, keepdims=True)
            if accum:
                ref = np.maximum(ref, B_np)
        elif op_type == "min":
            ref = A_data.min(axis=axes, keepdims=True)
            if accum:
                ref = np.minimum(ref, B_np)

        tvm.testing.assert_allclose(ref, B.numpy(), atol=1e-5)


@pytest.mark.parametrize("n_groups, n_warps", [(1, 1), (1, 4), (2, 8)])
@pytest.mark.parametrize("op_type", ["sum", "max", "min"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_local_view_complex(n_groups, n_warps, op_type, dtype, shuffle, accum):
    """Test view-based local reduction with wgmma layouts and optional shuffle."""
    if not shuffle and accum:
        pytest.skip("accum without shuffle is not supported in current implementation")
    dev = tvm.cuda(0)
    thread_cnt = 32
    NUM_COL = 128
    g_shape_a = (16 * n_warps, NUM_COL)
    g_shape_b = (16 * n_warps, 4)
    g_layout_a = TileLayout(S[g_shape_a])
    g_layout_b = TileLayout(S[g_shape_b])
    acc_shape, red_shape = (16, NUM_COL), (16, 4)

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape_a, dtype, layout=g_layout_a)
        B = Tx.match_buffer(B_ptr, g_shape_b, dtype, layout=g_layout_b)

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        wg_id = Tx.warpgroup_id([n_groups])
        warp_id_in_wg = Tx.warp_id_in_wg([n_warps // n_groups])
        lane_id = Tx.lane_id([thread_cnt])

        with Tx.thread():
                    # acc layout
            atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
            warp_layout = Tx.TileLayout(Tx.S[(8, 4) : (4@laneid, 1@laneid)])
            warp_atom = atom.tile(warp_layout, (8, 4), (1, 2))
            tile = Tx.TileLayout(Tx.S[(2, NUM_COL // 8) : (1, 2)])
            acc_layout = warp_atom.tile(tile, (2, NUM_COL // 8), (8, 8))
            acc = Tx.alloc_buffer(
                [2, NUM_COL // 4],
                dtype=dtype,
                scope="local",
                layout=atom.tile(tile, (2, NUM_COL // 8), (1, 2)),
            )

                    # red layout
            red_atom = Tx.TileLayout(Tx.S[(1, 1) : (1, 1)])
            red_warp_atom = red_atom.tile(warp_layout, (8, 4), (1, 1))
            red_tile = Tx.TileLayout(Tx.S[(2, 1) : (1, 1)])
            red_layout = red_warp_atom.tile(red_tile, (2, 1), (8, 4))
            red = Tx.alloc_buffer(
                [2],
                dtype=dtype,
                scope="local",
                layout=red_atom.tile(red_tile, (2, 1), (1, 1)),
            )

                    # Load A into acc
            with Tx.thread():
                for i in Tx.serial(NUM_COL // 8):
                    for j in Tx.unroll(2):
                        for vec in Tx.vectorized(2):
                            acc[j, i * 2 + vec] = A[
                                wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                i * 8 + lane_id % 4 * 2 + vec,
                            ]

                    # Pre-load B into red for accumulation
            if accum:
                with Tx.thread():
                    for i in Tx.unroll(2):
                        red[i] = B[
                            wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4,
                            lane_id % 4,
                        ]

                    # Reduce
            with Tx.warp():
                acc_view = acc.view(*acc_shape, layout=acc_layout)
                red_view = red.view(*red_shape, layout=red_layout)
                if op_type == "sum":
                    Tx.sum(red_view, acc_view, thread_reduce=shuffle, accum=accum)
                elif op_type == "max":
                    Tx.max(red_view, acc_view, thread_reduce=shuffle, accum=accum)
                elif op_type == "min":
                    Tx.min(red_view, acc_view, thread_reduce=shuffle, accum=accum)
                        # perform an additional shuffle step if not shuffled above
                if not shuffle:
                    if op_type == "sum":
                        Tx.sum(red_view, red_view, thread_reduce=True)
                    elif op_type == "max":
                        Tx.max(red_view, red_view, thread_reduce=True)
                    elif op_type == "min":
                        Tx.min(red_view, red_view, thread_reduce=True)
                    # Write red into B
            with Tx.thread():
                for i in Tx.unroll(2):
                    B[wg_id * 64 + warp_id_in_wg * 16 + i * 8 + lane_id // 4, lane_id % 4] = (
                        red[i]
                    )

        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape_a).astype(dtype)
        if accum:
            B_np = np.random.rand(*g_shape_b).astype(dtype) * 0.5
        else:
            B_np = np.zeros(g_shape_b, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np.copy(), dev)
        mod(A, B)

        if op_type == "sum":
            row_reduce = A_np.sum(axis=-1)
            if accum:
                B_ref = np.tile(row_reduce[:, np.newaxis], (1, 4)) + B_np
            else:
                B_ref = np.tile(row_reduce[:, np.newaxis], (1, 4))
        elif op_type == "max":
            row_reduce = A_np.max(axis=-1)
            if accum:
                B_ref = np.maximum(np.tile(row_reduce[:, np.newaxis], (1, 4)), B_np)
            else:
                B_ref = np.tile(row_reduce[:, np.newaxis], (1, 4))
        elif op_type == "min":
            row_reduce = A_np.min(axis=-1)
            if accum:
                B_ref = np.minimum(np.tile(row_reduce[:, np.newaxis], (1, 4)), B_np)
            else:
                B_ref = np.tile(row_reduce[:, np.newaxis], (1, 4))
        else:
            raise ValueError(f"Unsupported op_type: {op_type}")

        atol = 1e-5 if dtype == "float32" else 2e-1
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)


@pytest.mark.parametrize("reduction_len", [8, 16, 64, 128, 256, 7, 10, 15, 100])
@pytest.mark.parametrize("op_type", ["max", "min"])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_local_optimized_3input_maxmin(reduction_len, op_type, accum):
    """Test thread-level local buffer reduction with 3-input max/min PTX intrinsics."""
    dev = tvm.cuda(0)
    dtype = "float32"

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [reduction_len], dtype, layout=TileLayout(S[reduction_len]))
        B = Tx.match_buffer(B_ptr, [1], dtype, layout=TileLayout(S[1]))

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        _tid = Tx.thread_id([1])

        with Tx.thread():
            A_local = Tx.alloc_buffer([reduction_len], dtype, scope="local")
            B_local = Tx.alloc_buffer([1], dtype, scope="local")

                    # Load from global to local
            for i in Tx.serial(reduction_len):
                A_local[i] = A[i]

                    # Initialize B_local for accum test
            if accum:
                B_local[0] = B[0]

                    # Thread-level reduction
            if op_type == "max":
                Tx.max(B_local, A_local, accum=accum)
            elif op_type == "min":
                Tx.min(B_local, A_local, accum=accum)

                    # Store result to global
            B[0] = B_local[0]
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(reduction_len).astype(dtype)

        if accum:
            B_np = np.array([0.5], dtype=dtype)
        else:
            B_np = np.zeros(1, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        if op_type == "max":
            if accum:
                B_ref = max(A_np.max(), 0.5)
            else:
                B_ref = A_np.max()
        elif op_type == "min":
            if accum:
                B_ref = min(A_np.min(), 0.5)
            else:
                B_ref = A_np.min()

        tvm.testing.assert_allclose(B_ref, B.numpy()[0], atol=1e-5)


@pytest.mark.parametrize("reduction_len", [8, 16, 64, 128, 256, 9, 17, 63, 65, 100])
@pytest.mark.parametrize("accum", [False, True])
def test_reduction_local_optimized_packed_add_sum(reduction_len, accum):
    """Test thread-level sum reduction using packed add with add.f32x2 PTX instruction."""
    dev = tvm.cuda(0)
    dtype = "float32"

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [reduction_len], dtype, layout=TileLayout(S[reduction_len]))
        B = Tx.match_buffer(B_ptr, [1], dtype, layout=TileLayout(S[1]))

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        _tid = Tx.thread_id([1])

        with Tx.thread():
            A_local = Tx.alloc_buffer([reduction_len], dtype, scope="local")
            B_local = Tx.alloc_buffer([1], dtype, scope="local")

                    # Load from global to local
            for i in Tx.serial(reduction_len):
                A_local[i] = A[i]

                    # Initialize B_local for accum test
            if accum:
                B_local[0] = B[0]

                    # Thread-level sum reduction
            Tx.sum(B_local, A_local, accum=accum)

                    # Store result to global
            B[0] = B_local[0]
        # fmt: on

        # Use sm_100a target for packed add sum dispatch
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(reduction_len).astype(dtype)

        if accum:
            B_np = np.array([0.5], dtype=dtype)
        else:
            B_np = np.zeros(1, dtype=dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        if accum:
            B_ref = A_np.sum() + 0.5
        else:
            B_ref = A_np.sum()

        # Use larger tolerance due to rounding differences from packed add (add.rz.ftz.f32x2)
        tvm.testing.assert_allclose(B_ref, B.numpy()[0], atol=1e-4)


@pytest.mark.parametrize("op_type", ["sum", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_reduction_op_warp_shuffle(op_type, dtype):
    """Test warp-scope shuffle reduce with laneid shard→replica layout pattern.

    Case A: full warp reduce (32 lanes → 1 value, replicated to all lanes).
    """
    dev = tvm.cuda(0)
    N = 32
    g_shape = (N,)
    g_layout = TileLayout(S[N])

    # src layout: 32 elements sharded across 32 lanes
    src_layout = TileLayout(S[N : 1 @ laneid])
    # dst layout: 1 element replicated across 32 lanes
    dst_layout = TileLayout(S[1:1] + R[N : 1 @ laneid])

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        B = Tx.match_buffer(B_ptr, g_shape, dtype, layout=g_layout)

        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])

        with Tx.thread():
            src_local = Tx.alloc_buffer([1], dtype, scope="local")
            dst_local = Tx.alloc_buffer([1], dtype, scope="local")

            with Tx.thread():
                src_local[0] = A[lane_id]

            with Tx.warp():
                src_view = src_local.view(N, layout=src_layout)
                dst_view = dst_local.view(1, layout=dst_layout)
                if op_type == "sum":
                    Tx.sum(dst_view, src_view)
                elif op_type == "max":
                    Tx.max(dst_view, src_view)

            with Tx.thread():
                B[lane_id] = dst_local[0]
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(N).astype(dtype)
        B_np = np.zeros(N, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        if op_type == "sum":
            ref_val = A_np.astype("float64").sum()
        elif op_type == "max":
            ref_val = A_np.max()

        B_ref = np.full(N, ref_val, dtype=dtype)
        atol = 1e-4 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)


@pytest.mark.parametrize("op_type", ["sum", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_reduction_op_warp_shuffle_multi_elem(op_type, dtype):
    """Test warp-scope shuffle reduce with multiple elements per thread.

    Each thread holds 4 elements, reduce across 32 lanes for each element group.
    """
    dev = tvm.cuda(0)
    ELEMS_PER_THREAD = 4
    N_LANES = 32
    TOTAL = ELEMS_PER_THREAD * N_LANES  # 128
    g_shape = (TOTAL,)
    g_layout = TileLayout(S[TOTAL])

    # src: 32 lanes with 4 elements each; layout S[(32, 4) : (1@laneid, 1)]
    # element (i, j) → lane i, local j → thread k holds [4k, 4k+1, 4k+2, 4k+3]
    src_layout = TileLayout(S[(N_LANES, ELEMS_PER_THREAD) : (1 @ laneid, 1)])
    # dst: 4 elements per thread, replicated across 32 lanes
    dst_layout = TileLayout(S[ELEMS_PER_THREAD:1] + R[N_LANES : 1 @ laneid])

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, g_shape, dtype, layout=g_layout)
        dst_lay = TileLayout(S[ELEMS_PER_THREAD])
        B = Tx.match_buffer(B_ptr, [ELEMS_PER_THREAD], dtype, layout=dst_lay)

        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])

        with Tx.thread():
            src_local = Tx.alloc_buffer([ELEMS_PER_THREAD], dtype, scope="local")
            dst_local = Tx.alloc_buffer([ELEMS_PER_THREAD], dtype, scope="local")

            with Tx.thread():
                for i in Tx.serial(ELEMS_PER_THREAD):
                    src_local[i] = A[lane_id * ELEMS_PER_THREAD + i]

            with Tx.warp():
                src_view = src_local.view(TOTAL, layout=src_layout)
                dst_view = dst_local.view(ELEMS_PER_THREAD, layout=dst_layout)
                if op_type == "sum":
                    Tx.sum(dst_view, src_view)
                elif op_type == "max":
                    Tx.max(dst_view, src_view)

            with Tx.thread():
                for i in Tx.serial(ELEMS_PER_THREAD):
                    B[i] = dst_local[i]
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(TOTAL).astype(dtype)
        B_np = np.zeros(ELEMS_PER_THREAD, dtype=dtype)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # Each group of 4 elements: element j is sum/max of A[j], A[j+4], A[j+8], ..., A[j+124]
        A_reshaped = A_np.reshape(N_LANES, ELEMS_PER_THREAD)
        if op_type == "sum":
            B_ref = A_reshaped.astype("float64").sum(axis=0).astype(dtype)
        elif op_type == "max":
            B_ref = A_reshaped.max(axis=0)

        atol = 1e-4 if dtype == "float32" else 1e-1
        tvm.testing.assert_allclose(B_ref, B.numpy(), atol=atol)


def test_reduction_warp_shuffle_multi_warp_loop():
    """Test intra-warp + cross-warp reduction via Tx.sum in a for loop with multiple warps.

    Validates the scope alternation pattern (thread → warp → thread) inside a loop,
    which is needed for replacing manual warp shuffle reductions in tirx-kernels.
    """
    dev = tvm.cuda(0)
    BDX = 32
    BDY = 4
    N = BDX * BDY  # 128
    N_ITER = 3

    src_layout = TileLayout(S[BDX : 1 @ laneid])
    dst_layout = TileLayout(S[1:1] + R[BDX : 1 @ laneid])

    # fmt: off
    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, [N_ITER, N], "float32", scope="global")
        B = Tx.match_buffer(B_ptr, [N_ITER], "float32", scope="global")

        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        ty = Tx.warp_id([BDY])
        tx = Tx.lane_id([BDX])
        thread_id = Tx.meta_var(ty * BDX + tx)

        with Tx.cta():
            pool = Tx.SMEMPool()
            sum_smem = pool.alloc([BDY], "float32")
            pool.commit()

            with Tx.thread():
                partial_buf = Tx.alloc_buffer([1], "float32", scope="local")
                result_buf = Tx.alloc_buffer([1], "float32", scope="local")
                cross_buf = Tx.alloc_buffer([1], "float32", scope="local")
                cross_res = Tx.alloc_buffer([1], "float32", scope="local")

                for it in Tx.serial(N_ITER):
                            # Phase 1: each thread loads its value
                    with Tx.thread():
                        partial_buf[0] = A[it, thread_id]

                            # Phase 2: intra-warp reduction
                    with Tx.warp():
                        src_v = partial_buf.view(BDX, layout=src_layout)
                        dst_v = result_buf.view(1, layout=dst_layout)
                        Tx.sum(dst_v, src_v)

                            # Phase 3: write per-warp result to smem
                    with Tx.thread():
                        sum_smem[ty] = result_buf[0]
                    Tx.cuda.cta_sync()

                            # Phase 4: cross-warp reduction (warp 0 only)
                    if ty == 0:
                        with Tx.thread():
                            if tx < BDY:
                                cross_buf[0] = sum_smem[tx]
                            else:
                                cross_buf[0] = Tx.float32(0)
                        with Tx.warp():
                            cs = cross_buf.view(BDX, layout=src_layout)
                            cd = cross_res.view(1, layout=dst_layout)
                            Tx.sum(cd, cs)
                        with Tx.thread():
                            sum_smem[0] = cross_res[0]
                    Tx.cuda.cta_sync()

                            # Phase 5: one thread writes result to global
                    with Tx.thread():
                        if tx == 0:
                            if ty == 0:
                                B[it] = sum_smem[0]
                    Tx.cuda.cta_sync()
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(42)
        A_np = np.random.rand(N_ITER, N).astype("float32")
        B_np = np.zeros(N_ITER, dtype="float32")
        A_dev = tvm.runtime.tensor(A_np, dev)
        B_dev = tvm.runtime.tensor(B_np, dev)
        mod(A_dev, B_dev)

        # Each iteration: sum across all N threads
        B_ref = A_np.astype("float64").sum(axis=1).astype("float32")
        tvm.testing.assert_allclose(B_ref, B_dev.numpy(), atol=1e-3)


@pytest.mark.parametrize("op_name", ["sum", "max"])
def test_reduction_warpgroup_wg_local_layout(op_name):
    rows, cols = 128, 16
    dtype = "float32"
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (rows, cols), dtype, layout=TileLayout(S[(rows, cols)]))
        B = Tx.match_buffer(B_ptr, (rows, 1), dtype, layout=TileLayout(S[(rows, 1)]))

        Tx.device_entry()
        _bx = Tx.cta_id([1])
        wg_id = Tx.warpgroup_id([1])
        tid = Tx.thread_id_in_wg([rows])

        src = Tx.alloc_buffer((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))
        dst = Tx.alloc_buffer((rows, 1), dtype, scope="local", layout=wg_local_layout(1))

        with Tx.thread():
            src_local = src.local(cols)
            for i in Tx.serial(cols):
                src_local[i] = A[tid, i]

        with Tx.warpgroup():
            if op_name == "sum":
                Tx.sum(dst, src, axes=[-1], accum=False)
            else:
                Tx.max(dst, src, axes=[-1], accum=False)

        with Tx.thread():
            dst_local = dst.local(1)
            B[tid, 0] = dst_local[0]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.zeros((rows, 1), dtype=dtype)
        A_dev = tvm.runtime.tensor(A_np, dev)
        B_dev = tvm.runtime.tensor(B_np, dev)

        mod = tvm.IRModule({"main": test_func})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        mod(A_dev, B_dev)

        if op_name == "sum":
            B_ref = A_np.sum(axis=1, keepdims=True)
        else:
            B_ref = A_np.max(axis=1, keepdims=True)
        tvm.testing.assert_allclose(B_ref, B_dev.numpy(), atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
