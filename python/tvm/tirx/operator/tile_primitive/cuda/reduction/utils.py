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

"""Shared helpers for reduction operator dispatches on CUDA targets."""

import functools
import math
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ...common import ReduceOpType
from ..common import match_scope

reduce_op_table = {
    ReduceOpType.SUM: lambda a, b: a + b,
    ReduceOpType.MAX: T.max,
    ReduceOpType.MIN: T.min,
}


def reduce_default_value_table(dtype):
    return {
        ReduceOpType.SUM: 0.0,
        ReduceOpType.MAX: T.min_value(dtype),
        ReduceOpType.MIN: T.max_value(dtype),
    }


def _reduction_args(
    op: TilePrimitiveCall,
) -> tuple[BufferRegion, BufferRegion, tuple[int, ...], bool, dict]:
    """Parse ReduceOp -> (dst, src, reduce_axes, accum, config)."""
    op = TilePrimitiveCall.downcast(op)
    dst = op.output
    src = op.input
    reduce_axes = tuple(int(a) for a in op.reduce_axes)
    accum = op.accum
    config = op.config
    return dst, src, reduce_axes, accum, config


def _match_reduction_storage_scope(
    op: TilePrimitiveCall, sctx: DispatchContext, expected_scope: list[str]
) -> tuple[bool, str | None]:
    """Check that dst and src scopes match one of the expected patterns."""
    op = TilePrimitiveCall.downcast(op)
    dst_scope = op.output.buffer.scope()
    src_scope = op.input.buffer.scope()

    ok = any(match_scope(dst_scope, p) and match_scope(src_scope, p) for p in expected_scope)
    msg = f"storage scope mismatch: dst {dst_scope}, src {src_scope}; expected {expected_scope}"
    return (ok, None if ok else msg)


def _analyze_axes(src_ndim: int, reduce_axes: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Normalize negative axes -> (reduce_dim_set, spatial_dim_list)."""
    reduce_dims = set()
    for ax in reduce_axes:
        a = ax if ax >= 0 else ax + src_ndim
        assert 0 <= a < src_ndim, f"reduce axis {ax} out of range for ndim={src_ndim}"
        reduce_dims.add(a)
    spatial_dims = [d for d in range(src_ndim) if d not in reduce_dims]
    return sorted(reduce_dims), spatial_dims


def _analyze_layout_dims(layout, shape):
    """layout.group(shape) -> decompose each dim into thread/local iters.

    Returns list of per-dim (thread_extent, local_extent, thread_strides):
        thread_extent = product of thread iter extents in this dim
        local_extent  = product of local iter extents in this dim
        thread_strides = list of (stride, extent) for thread iters in this dim
    """
    grouped, seps = layout.group(list(shape))
    result = []
    for d in range(len(shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        thread_extent = 1
        local_extent = 1
        thread_strides = []
        for s_idx in shard_range:
            it = grouped.shard[s_idx]
            if it.axis.is_thread():
                thread_extent *= it.extent
                thread_strides.append((it.stride, it.extent))
            else:
                local_extent *= it.extent
        result.append((thread_extent, local_extent, thread_strides))
    return result


def _compute_shuffle_masks(dim_info, reduce_dims: set[int]) -> list[int]:
    """From reduction dims' thread iter (stride, extent) pairs, compute XOR masks.

    For each thread iter in a reduction dim:
        masks += [stride * 2^i for i in range(log2(extent))]
    Sorted ascending.
    """
    masks = []
    for d in reduce_dims:
        _, _, thread_strides = dim_info[d]
        for stride, extent in thread_strides:
            ext_int = int(extent) if hasattr(extent, "__int__") else extent
            n_bits = int(math.log2(ext_int))
            for i in range(n_bits):
                stride_int = int(stride) if hasattr(stride, "__int__") else stride
                masks.append(stride_int * (1 << i))
    masks.sort()
    return masks


def _build_local_dim_map(layout, buffer_shape):
    """Map original dim index to position in get_local_region output (None if pure-thread)."""
    grouped, seps = layout.group(list(buffer_shape))
    dim_map = {}
    local_pos = 0
    for d in range(len(buffer_shape)):
        shard_range = list(range(seps[d], seps[d + 1]))
        has_local = any(not grouped.shard[s].axis.is_thread() for s in shard_range)
        if has_local:
            dim_map[d] = local_pos
            local_pos += 1
        else:
            dim_map[d] = None
    return dim_map


def _validate_reduction_layout(
    src_layout, dst_layout, src_shape, dst_shape, reduce_dims: list[int]
) -> tuple[bool, str | None]:
    """Validate that spatial dims of src/dst have matching thread+local structure,
    and that reduction dims in dst have local_extent == 1.
    """
    src_dim_info = _analyze_layout_dims(src_layout, src_shape)
    dst_dim_info = _analyze_layout_dims(dst_layout, dst_shape)
    analyzer = Analyzer()

    # Spatial dims: src/dst must match in both thread and local extents.
    # Reduce dims: src/dst thread extent must match, and dst local extent must be 1.

    # get expected simplified dst layout
    expected_dst_dim = []
    for src_idx in range(len(src_shape)):
        if analyzer.can_prove_equal(src_dim_info[src_idx][0], 1) and analyzer.can_prove_equal(
            src_dim_info[src_idx][1], 1
        ):
            continue  # skip if extent=1
        if src_idx in reduce_dims:  # reduce dims
            if not analyzer.can_prove_equal(src_dim_info[src_idx][0], 1):
                expected_dst_dim.append((src_dim_info[src_idx][0], 1))
        else:  # spatial dims
            expected_dst_dim.append((src_dim_info[src_idx][0], src_dim_info[src_idx][1]))

    # check dst layout
    check_idx = 0
    for dst_idx in range(len(dst_shape)):
        if analyzer.can_prove_equal(dst_dim_info[dst_idx][0], 1) and analyzer.can_prove_equal(
            dst_dim_info[dst_idx][1], 1
        ):
            continue
        if not (
            analyzer.can_prove_equal(dst_dim_info[dst_idx][0], expected_dst_dim[check_idx][0])
            and analyzer.can_prove_equal(dst_dim_info[dst_idx][1], expected_dst_dim[check_idx][1])
        ):
            return False, "mismatch dst/src layout for reduction"
        check_idx += 1
    if check_idx != len(expected_dst_dim):
        return False, "mismatch dst/src layout for reduction"
    return True, None


def build_src_indices(spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st):
    """Combine spatial and reduction indices into full src index tuple."""

    # Build index helpers that work with the explicit axis split
    def get_spatial_or_reduction_src_indices(spa_or_red_fused, is_spatial):
        dims = spatial_dims if is_spatial else reduce_dims
        spa_extents = [src_extent[d] for d in dims]
        indices = []
        rem = spa_or_red_fused
        for e in reversed(spa_extents):
            indices.append(rem % e)
            rem //= e
        indices.reverse()
        return [idx + src_st[d] for idx, d in zip(indices, dims)]

    spa_vals = get_spatial_or_reduction_src_indices(spa_fused, is_spatial=True)
    red_vals = get_spatial_or_reduction_src_indices(red_fused, is_spatial=False)
    full = [None] * len(src_extent)
    for i, d in enumerate(spatial_dims):
        full[d] = spa_vals[i]
    for i, d in enumerate(reduce_dims):
        full[d] = red_vals[i]
    return full


_REDUCE_OP_TO_STR = {ReduceOpType.SUM: "sum", ReduceOpType.MAX: "max", ReduceOpType.MIN: "min"}


def _dtype_ok(op: TilePrimitiveCall, sctx: DispatchContext, expected_dtype: str):
    op = TilePrimitiveCall.downcast(op)
    dtype = op.input.buffer.dtype
    ok = dtype == expected_dtype
    return (ok, None if ok else f"dtype {dtype} != {expected_dtype}")


def _reduction_len_ok(op: TilePrimitiveCall, sctx: DispatchContext, min_len: int):
    op = TilePrimitiveCall.downcast(op)
    src_extent = [r.extent for r in op.input.region]
    reduction_len = functools.reduce(operator.mul, src_extent, 1)
    ok = reduction_len >= min_len
    return (ok, None if ok else f"reduction_len {reduction_len} < {min_len}")


def _dst_len_ok(op: TilePrimitiveCall, sctx: DispatchContext, expected_len: int):
    op = TilePrimitiveCall.downcast(op)
    dst_extent = [r.extent for r in op.output.region]
    dst_len = functools.reduce(operator.mul, dst_extent, 1)
    ok = dst_len == expected_len
    return (ok, None if ok else f"dst_len {dst_len} != {expected_len}")


def _src_ndim_ok(op: TilePrimitiveCall, sctx: DispatchContext, expected_ndim: int):
    op = TilePrimitiveCall.downcast(op)
    src_extent = [r.extent for r in op.input.region]
    ok = len(src_extent) == expected_ndim
    return (ok, None if ok else f"src ndim {len(src_extent)} != {expected_ndim}")


def _local_scope_match(op: TilePrimitiveCall, sctx: DispatchContext):
    op = TilePrimitiveCall.downcast(op)
    src, dst = op.input.buffer, op.output.buffer
    ok = all(
        [src.scope() == "local", dst.scope() == "local", src.dtype == dst.dtype, sctx.is_cuda()]
    )
    if not ok:
        return (False, "src/dst must be local scope with matching dtype on CUDA")
    return (True, None)
