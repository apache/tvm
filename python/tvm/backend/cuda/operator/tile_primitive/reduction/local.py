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

"""CUDA reduction operator dispatch: local-memory variant.

Registered ops: sum, max, min.

When: dst and src are both local-scope buffers with matching dtype, on CUDA.

(A) Thread scope -- sequential per-element reduction
    (_emit_reduction_local_thread_wise):

Before:
    Tx.sum(B_local[0:2, 0:3], A_local[0:2, 0:3, 0:4], [-1], False)

After (scheduled PrimFunc, spatial_len=6, reduction_len=4):
    for spa in range(6):
        B_local[spa] = T.float32(0.0)                      # init (skipped if accum)
        for red in range(4):
            B_local[spa] = B_local[spa] + A_local[spa * 4 + red]

(B) Warp/Warpgroup scope -- layout-driven reduction
    (_emit_reduction_local_view):
    Requires TileLayout with valid thread-partition. Decomposes layout to
    identify thread-local elements, then optionally shuffles partial sums.

    thread_reduce=False: local-only, no shuffle (warp and warpgroup).
    thread_reduce=True: local reduction + cross-thread shfl_xor steps (warp only).
    accum=True + shuffle: saves old dst before reduce+shuffle, combines after (warp only).

Before:
    Tx.warp.sum(red_view[0:16, 0:4], acc_view[0:16, 0:128], [-1], False,
                   thread_reduce=True)

After (scheduled PrimFunc, local_total=2, local_red=32, 2 shuffle steps):
    src_local = acc_view.view(64)
    dst_local = red_view.view(2)
    for spa in range(2):
        dst_local[spa] = T.float32(0.0)
        for red in range(32):
            dst_local[spa] = dst_local[spa] + src_local[...]
        dst_local[spa] = dst_local[spa] + shfl_xor(..., 1, 32, 32)
        dst_local[spa] = dst_local[spa] + shfl_xor(..., 2, 32, 32)
"""

import functools
import operator
from typing import Any

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc
from tvm.tirx.layout import TileLayout, laneid
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.operator.tile_primitive.common import ReduceOpType
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import get_indices, get_st_extent
from ..layout_utils import get_local_region, get_sublayout_from_region
from .utils import (
    _REDUCE_OP_TO_STR,
    _analyze_axes,
    _analyze_layout_dims,
    _build_local_dim_map,
    _compute_shuffle_masks,
    _match_reduction_storage_scope,
    _reduction_args,
    _validate_reduction_layout,
    reduce_default_value_table,
    reduce_op_table,
)


def _analyze_shuffle_reduce(src_layout, dst_layout):
    """Analyze src/dst layouts for laneid shard->replica reduce pattern.

    Returns (reduce_width, local_elems) if the pattern matches, or None.
    - reduce_width: number of lanes participating in each group's reduction
    - local_elems: per-thread element count (product of non-laneid shard extents)
    """
    if src_layout.is_swizzle() or dst_layout.is_swizzle():
        return None

    src_canon = src_layout.canonicalize()
    dst_canon = dst_layout.canonicalize()

    # Extract laneid iters from shard and replica
    src_laneid_shard = [it for it in src_canon.shard if it.axis == laneid]
    dst_laneid_replica = [it for it in dst_canon.replica if it.axis == laneid]

    # src shard must contain laneid (data distributed across lanes)
    if not src_laneid_shard:
        return None
    # dst replica must contain laneid (result broadcast to lanes)
    if not dst_laneid_replica:
        return None

    # laneid span must be 32 (full warp)
    src_laneid_span = 1 + sum(abs(int(it.stride)) * (int(it.extent) - 1) for it in src_laneid_shard)
    if src_laneid_span != 32:
        return None

    reduce_width = functools.reduce(operator.mul, [int(it.extent) for it in dst_laneid_replica], 1)
    if reduce_width <= 0 or reduce_width > 32 or (reduce_width & (reduce_width - 1)) != 0:
        return None  # must be power of 2

    # local_elems = product of non-laneid shard extents in src
    src_non_laneid = [it for it in src_canon.shard if it.axis != laneid]
    local_elems = functools.reduce(operator.mul, [int(it.extent) for it in src_non_laneid], 1)

    return reduce_width, local_elems


def _gen_warp_shuffle_reduce(src, dst, reduce_width, local_elems, accum, op_type, init_value):
    """Generate warp shuffle reduce codegen for laneid shard->replica pattern.

    Unified for both full warp (reduce_width=32) and partial warp (e.g. reduce_width=8).
    """
    is_same_buffer = src.same_as(dst)
    op_str = _REDUCE_OP_TO_STR[op_type]

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        src_local = src.local(local_elems)
        dst_local = dst.local(local_elems)
        for k in T.serial(local_elems):
            if not is_same_buffer:
                dst_local[k] = src_local[k]
            dst_local[k] = T.cuda.warp_reduce(dst_local[k], op_str, reduce_width)
    # fmt: on

    return impl


def validate_reduction_local(
    op: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Validate reduction in local memory."""
    op = TilePrimitiveCall.downcast(op)
    dst_br, src_br = op.output, op.input
    dst, src = dst_br.buffer, src_br.buffer

    if not (src.scope() == "local" and dst.scope() == "local" and sctx.is_target("cuda")):
        return False, "expected local scope and CUDA target"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"

    if sctx.is_thread:
        return True, None  # thread-wise reduction
    elif sctx.scope_kind in ["warp", "warpgroup"]:
        if not sctx.is_warp and op.config.get("thread_reduce", False):
            return (
                False,
                "thread_reduce=True is only supported in warp scope; "
                "warpgroup local reduction is thread-local only",
            )
        # VIEW: need layouts and layout analysis
        if not (src.layout and dst.layout):
            return False, "layouts required for view-based local reduction"
        if not (isinstance(src.layout, TileLayout) and isinstance(dst.layout, TileLayout)):
            return False, "TileLayout required for view-based local reduction"
        if src.layout.is_swizzle() or dst.layout.is_swizzle():
            return False, "swizzle layout unsupported for local reduction"

        analyzer = Analyzer()

        # Validate get_local_region succeeds for both
        src_st, src_extent = get_st_extent(src_br)
        dst_st, dst_extent = get_st_extent(dst_br)

        if sctx.is_warp:
            # Check for laneid shard->replica shuffle reduce pattern first.
            # This pattern has laneid in dst replica (broadcast), which the
            # general validation below would reject.
            shuffle_info = _analyze_shuffle_reduce(src.layout, dst.layout)
            if shuffle_info is not None:
                return True, None

        for layout, buf, st, ext, name in [
            (src.layout, src, src_st, src_extent, "src"),
            (dst.layout, dst, dst_st, dst_extent, "dst"),
        ]:
            for it in layout.shard:
                if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
                    return False, f"thread dim with zero stride in {name}"
            replica = getattr(layout, "replica", None) or []
            if any(it.axis.is_thread() for it in replica):
                return False, f"thread axis in replica for {name}"
            if get_local_region(layout, list(buf.shape), st, ext) is None:
                return False, f"get_local_region failed for {name}"

        # Validate layout compatibility
        # Spatial dims match, reduce dims in dst have local_extent==1
        reduce_axes = tuple(int(a) for a in op.reduce_axes)
        src_ndim = len(src_br.region)
        try:
            reduce_dims, _ = _analyze_axes(src_ndim, reduce_axes)
        except AssertionError as e:
            return False, str(e)
        src_sliced = get_sublayout_from_region(src.layout, src.shape, src_st, src_extent)
        dst_sliced = get_sublayout_from_region(dst.layout, dst.shape, dst_st, dst_extent)
        ok, msg = _validate_reduction_layout(
            src_sliced, dst_sliced, list(src_extent), list(dst_extent), reduce_dims
        )
        return ok, msg
    else:
        return False, f"unsupported exec_scope {sctx.scope_kind} for local reduction"


def _emit_reduction_local_thread_wise(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    dtype = src.dtype
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    src_ndim = len(src_extent)
    spa_extents = [src_extent[d] for d in spatial_dims]
    red_extents = [src_extent[d] for d in reduce_dims]
    spatial_len = functools.reduce(operator.mul, spa_extents, 1)
    reduction_len = functools.reduce(operator.mul, red_extents, 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    def get_src_indices(spa_fused, red_fused):
        spa_indices = []
        rem = spa_fused
        for e in reversed(spa_extents):
            spa_indices.append(rem % e)
            rem //= e
        spa_indices.reverse()

        red_indices = []
        rem = red_fused
        for e in reversed(red_extents):
            red_indices.append(rem % e)
            rem //= e
        red_indices.reverse()

        full = [None] * src_ndim
        for i, d in enumerate(spatial_dims):
            full[d] = spa_indices[i] + src_st[d]
        for i, d in enumerate(reduce_dims):
            full[d] = red_indices[i] + src_st[d]
        return full

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        for spa in T.serial(spatial_len):
            dst_idx = T.meta_var(get_indices(spa, dst_st, dst_extent))
            if not accum:
                dst[tuple(dst_idx)] = init_value
            for red in T.serial(reduction_len):
                src_idx = T.meta_var(get_src_indices(spa, red))
                dst[tuple(dst_idx)] = op_func(dst[tuple(dst_idx)], src[tuple(src_idx)])
    # fmt: on

    return impl


def _emit_reduction_local_view(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    config: dict[str, Any],
    reduce_dims: set[int],
    spatial_dims: list[int],
    src_local_info,
    dst_local_info,
    shuffle_masks: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    dtype = src.dtype

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    src_local_shape, src_local_st, src_local_ext = src_local_info
    dst_local_shape, dst_local_st, dst_local_ext = dst_local_info

    # Build maps from original dim index to position in get_local_region output
    src_dim_map = _build_local_dim_map(src.layout, list(src.shape))
    dst_dim_map = _build_local_dim_map(dst.layout, list(dst.shape))

    # Only include reduction dims that have local parts in src
    src_ndim = len(src_br.region)
    reduce_local_dims = [d for d in reduce_dims if src_dim_map[d] is not None]
    reduction_local_ext = [src_local_ext[src_dim_map[d]] for d in reduce_local_dims]
    reduction_local_st = [src_local_st[src_dim_map[d]] for d in reduce_local_dims]

    reduction_local_total = functools.reduce(operator.mul, reduction_local_ext, 1)
    dst_local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    def _get_src_local_index(dst_fused, red_fused):
        """Compute src local multi-dim index from dst fused index and reduction fused index."""
        dst_indices = get_indices(dst_fused, dst_local_st, dst_local_ext)
        red_indices = get_indices(red_fused, reduction_local_st, reduction_local_ext)

        # Interleave into src local indices (skipping pure-thread dims)
        src_local = []
        ri = 0
        for d in range(src_ndim):
            if src_dim_map[d] is None:
                continue  # pure-thread in src, not in src.local()
            if d in reduce_dims:
                src_local.append(red_indices[ri])
                ri += 1
            else:
                # Spatial dim: use corresponding dst local position
                src_local.append(dst_indices[dst_dim_map[d]])

        return src_local

    # is_same_buffer = src.same_as(dst)
    shuffle = bool(config.get("thread_reduce", False))
    in_place = dst.same_as(src)

    def shuffle_data(mask, dst_local, dst_idx):
        @T.inline
        def inner_shuffle(v, shuffle_mask):
            dst_local[tuple(dst_idx)] = op_func(
                v, T.tvm_warp_shuffle_xor(mask, v, shuffle_mask, 32, 32)
            )

        for i in range(len(shuffle_masks)):
            inner_shuffle(dst_local[tuple(dst_idx)], shuffle_masks[i])

    need_save_accum = accum and shuffle

    # fmt: off
    if need_save_accum:
        @T.prim_func(check_well_formed=False)
        def impl():
            src_local = src.local(*src_local_shape)
            dst_local = dst.local(*dst_local_shape)
            old_val = T.alloc_buffer([1], dtype, scope="local")

            for spa in T.serial(dst_local_total):
                dst_idx = T.meta_var(get_indices(spa, dst_local_st, dst_local_ext))
                old_val[0] = dst_local[tuple(dst_idx)]
                if not in_place:
                    dst_local[tuple(dst_idx)] = init_value
                    for red in T.serial(reduction_local_total):
                        src_idx = T.meta_var(_get_src_local_index(spa, red))
                        dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], src_local[tuple(src_idx)])  # noqa: E501
                if shuffle:
                    mask = T.tvm_warp_activemask()
                    shuffle_data(mask, dst_local, dst_idx)
                dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], old_val[0])
    else:
        @T.prim_func(check_well_formed=False)
        def impl():
            src_local = src.local(*src_local_shape)
            dst_local = dst.local(*dst_local_shape)

            for spa in T.serial(dst_local_total):
                dst_idx = T.meta_var(get_indices(spa, dst_local_st, dst_local_ext))
                if not in_place:
                    if not accum:
                        dst_local[tuple(dst_idx)] = init_value
                    for red in T.serial(reduction_local_total):
                        src_idx = T.meta_var(_get_src_local_index(spa, red))
                        dst_local[tuple(dst_idx)] = op_func(dst_local[tuple(dst_idx)], src_local[tuple(src_idx)])  # noqa: E501
                if shuffle:
                    mask = T.tvm_warp_activemask()
                    shuffle_data(mask, dst_local, dst_idx)
    # fmt: on

    return impl


def reduction_local_impl(
    op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext
) -> PrimFunc | None:
    dst_br, src_br, reduce_axes, accum, config = _reduction_args(op)
    src_ndim = len(src_br.region)
    reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)

    if sctx.is_thread:
        return _emit_reduction_local_thread_wise(
            dst_br, src_br, accum, op_type, reduce_dims, spatial_dims
        )
    elif sctx.scope_kind in ["warp", "warpgroup"]:
        src = src_br.buffer
        dst = dst_br.buffer

        if sctx.is_warp:
            # --- Try laneid shard->replica shuffle reduce ---
            shuffle_info = _analyze_shuffle_reduce(src.layout, dst.layout)
            if shuffle_info is not None:
                reduce_width, local_elems = shuffle_info
                if op_type not in _REDUCE_OP_TO_STR:
                    fail(f"unsupported reduce op: {op_type}")
                dtype = src.dtype
                init_value = reduce_default_value_table(dtype).get(op_type)
                return _gen_warp_shuffle_reduce(
                    src, dst, reduce_width, local_elems, accum, op_type, init_value
                )
        elif config.get("thread_reduce", False):
            fail(
                "thread_reduce=True is only supported in warp scope; "
                "warpgroup local reduction is thread-local only"
            )

        # --- Existing WGMMA layout path below ---
        src_st, src_extent = get_st_extent(src_br)
        dst_st, dst_extent = get_st_extent(dst_br)

        src_local_info = get_local_region(src.layout, list(src.shape), src_st, src_extent)
        dst_local_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_extent)
        assert src_local_info is not None and dst_local_info is not None

        src_dim_info = _analyze_layout_dims(src.layout, list(src.shape))
        shuffle_masks = (
            _compute_shuffle_masks(src_dim_info, reduce_dims)
            if config.get("thread_reduce", False)
            else []
        )

        return _emit_reduction_local_view(
            dst_br,
            src_br,
            accum,
            op_type,
            config,
            reduce_dims,
            spatial_dims,
            src_local_info,
            dst_local_info,
            shuffle_masks,
        )
    else:
        fail(f"unsupported exec_scope {sctx.scope_kind} for reduction_local_impl")


# ---------------------------------------------------------------------------
# Registration: local memory reduction (priority=10)
# ---------------------------------------------------------------------------

for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:

    @register_dispatch(
        op_name,
        "cuda",
        variant="local",
        priority=10,
        when=[
            predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["local"]),
            predicate("local_valid", validate_reduction_local),
        ],
    )
    def _local_dispatch(op: TilePrimitiveCall, sctx: DispatchContext, _op_type=op_type) -> PrimFunc:
        op = TilePrimitiveCall.downcast(op)
        return reduction_local_impl(op, _op_type, sctx)
