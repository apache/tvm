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

"""CUDA reduction operator dispatch: shared-memory variant.

Registered ops: sum, max, min.

When: dst and src are both shared-memory buffers, exec scope is one of
{cta, warpgroup, warp, thread}, threadIdx.x bound, reduce axes valid.

(A) CTA/warpgroup/warp scope -- adaptive-group shuffle tree
    (_emit_reduction_shared_cta):
    group_size = min(next_power_of_2(reduction_len), 32).
    Each group of threads reduces one spatial position via shfl_xor.

Before:
    Tx.cta.sum(B_smem[0:4], A_smem[0:4, 0:8], [-1], False)

After (scheduled PrimFunc, group_size=8, spatial_par=4):
    thread_data[0] = T.float32(0.0)
    thread_data[0] = thread_data[0] + A_smem[tid_in_scope]  # gather
    # log2(8) = 3 shuffle-xor steps with width=8
    thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 1, 8, 32)
    thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 2, 8, 32)
    thread_data[0] = thread_data[0] + shfl_xor(thread_data[0], 4, 8, 32)
    if tid_in_scope % 8 == 0:
        B_smem[tid_in_scope // 8] = thread_data[0]

(B) Thread scope -- sequential loop (_emit_reduction_shared_thread):

Before:
    if tid == 65:
        Tx.sum(B_smem[0:4], A_smem[0:4, 0:8], [-1], False)

After (scheduled PrimFunc):
    for spa in range(4):
        B_smem[spa] = T.float32(0.0)                       # init (skipped if accum)
        for red in range(8):
            B_smem[spa] = B_smem[spa] + A_smem[spa * 8 + red]
"""

import functools
import math
import operator

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.operator.tile_primitive.common import ReduceOpType
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import get_indices, get_st_extent, next_power_of_2
from .utils import (
    _analyze_axes,
    _match_reduction_storage_scope,
    _reduction_args,
    build_src_indices,
    reduce_default_value_table,
    reduce_op_table,
)


def validate_reduction_shared(
    op: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Validate reduction in shared memory."""
    if sctx.scope_kind not in ["cta", "warpgroup", "warp", "thread"]:
        return False, f"unsupported exec_scope {sctx.scope_kind} for shared reduction"

    op = TilePrimitiveCall.downcast(op)
    dst, src = op.output.buffer, op.input.buffer
    if not (src.scope().startswith("shared") and dst.scope().startswith("shared")):
        return False, "expected shared scope for both src and dst"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype} dst={dst.dtype}"

    if "threadIdx.x" not in sctx.launch_params:
        return False, "threadIdx.x not in launch_params"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return False, "multi-dimensional thread binding not supported for shared reduction"

    reduce_axes = tuple(int(a) for a in op.reduce_axes)
    src_region = op.input.region
    dst_region = op.output.region
    src_ndim = len(src_region)
    try:
        reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)
    except AssertionError as e:
        return False, str(e)

    # Validate dst shape matches spatial dims of src
    src_extent = [r.extent for r in src_region]
    dst_extent = [r.extent for r in dst_region]
    expected_dst_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    actual_dst_len = functools.reduce(operator.mul, dst_extent, 1)
    analyzer = Analyzer()
    if not analyzer.can_prove_equal(expected_dst_len, actual_dst_len):
        return (False, f"dst size {actual_dst_len} != expected spatial size {expected_dst_len}")

    return True, None


def _emit_reduction_shared_cta(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    exec_scope_name = sctx.scope_kind

    def get_thread_cnt():
        if exec_scope_name == "cta":
            return sctx.launch_params["threadIdx.x"].dom.extent
        elif exec_scope_name == "warpgroup":
            return 128
        elif exec_scope_name == "warp":
            return 32
        elif exec_scope_name == "thread":
            return 1

    thread_cnt = get_thread_cnt()
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    dtype = src.dtype

    # Compute spatial/reduction from the explicit axes
    spatial_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[d] for d in reduce_dims], 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    # Adaptive group size: nearest power-of-2 for reduction length, capped at warp size and thread count. # noqa: E501
    group_size = min(next_power_of_2(int(reduction_len)), 32, int(thread_cnt))
    group_size = max(group_size, 1)  # ensure at least 1
    n_shuffles = int(math.log2(group_size)) if group_size > 1 else 0
    spatial_par = int(thread_cnt) // group_size

    def get_tid_in_scope():
        tx_var = sctx.launch_params["threadIdx.x"].var
        if exec_scope_name == "cta":
            return tx_var
        elif exec_scope_name in ("warp", "warpgroup"):
            return tx_var % thread_cnt
        elif exec_scope_name == "thread":
            return 0

    def shuffle_data(thread_data):
        @T.inline
        def inner_shuffle(mask, v, shuffle_mask):
            v[0] = op_func(v[0], T.tvm_warp_shuffle_xor(mask, v[0], shuffle_mask, group_size, 32))

        if n_shuffles > 0:
            mask = T.tvm_warp_activemask()
            for i in range(n_shuffles):
                inner_shuffle(mask, thread_data, 1 << i)

    @T.inline
    def sync():
        if exec_scope_name == "cta":
            T.cuda.cta_sync()
        elif exec_scope_name == "warpgroup":
            T.cuda.warpgroup_sync(8)  # TODO: fix this hardcoded value
        elif exec_scope_name == "warp":
            T.cuda.warp_sync()
        elif exec_scope_name == "thread":
            pass

    # fmt: off
    @T.prim_func
    def impl():
        tid_in_scope = get_tid_in_scope()
        thread_data = T.alloc_buffer([1], dtype=dtype, scope="local")
        group_id = T.meta_var(T.floordiv(tid_in_scope, group_size))
        lane_in_grp = T.meta_var(tid_in_scope % group_size)
        for step in T.serial(T.ceildiv(spatial_len, spatial_par)):
            spa_fused = T.meta_var(step * spatial_par + group_id)
            if spa_fused < spatial_len:
                thread_data[0] = init_value
                for t in T.serial(T.ceildiv(reduction_len, group_size)):
                    red_fused = T.meta_var(t * group_size + lane_in_grp)
                    if red_fused < reduction_len:
                        src_indices = T.meta_var(build_src_indices(spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st))  # noqa: E501
                        thread_data[0] = op_func(thread_data[0], src[tuple(src_indices)])
                shuffle_data(thread_data)
                if lane_in_grp == 0:
                    dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
                    dst[tuple(dst_indices)] = T.if_then_else(T.bool(accum), op_func(dst[tuple(dst_indices)], thread_data[0]), thread_data[0])  # noqa: E501

        sync()
    # fmt: on

    return impl


def _emit_reduction_shared_thread(
    dst_br: BufferRegion,
    src_br: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
    reduce_dims: list[int],
    spatial_dims: list[int],
) -> PrimFunc:
    dst, src = dst_br.buffer, src_br.buffer
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    dtype = src.dtype

    # Compute spatial/reduction from the explicit axes
    spatial_len = functools.reduce(operator.mul, [src_extent[d] for d in spatial_dims], 1)
    reduction_len = functools.reduce(operator.mul, [src_extent[d] for d in reduce_dims], 1)

    op_func = reduce_op_table.get(reduce_op)
    assert op_func is not None
    init_value = reduce_default_value_table(dtype).get(reduce_op)

    @T.prim_func
    def impl():
        for spa_fused in T.serial(spatial_len):
            dst_indices = T.meta_var(get_indices(spa_fused, dst_st, dst_extent))
            if not accum:
                dst[tuple(dst_indices)] = init_value
            for red_fused in T.serial(reduction_len):
                src_indices = T.meta_var(
                    build_src_indices(
                        spa_fused, red_fused, spatial_dims, reduce_dims, src_extent, src_st
                    )
                )
                dst[tuple(dst_indices)] = op_func(dst[tuple(dst_indices)], src[tuple(src_indices)])

    return impl


def reduction_shared_impl(
    op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext
) -> PrimFunc | None:
    dst_br, src_br, reduce_axes, accum, config = _reduction_args(op)
    src_ndim = len(src_br.region)
    reduce_dims, spatial_dims = _analyze_axes(src_ndim, reduce_axes)
    if sctx.scope_kind in ["cta", "warpgroup", "warp"]:
        return _emit_reduction_shared_cta(
            dst_br, src_br, accum, op_type, sctx, reduce_dims, spatial_dims
        )
    elif sctx.is_thread:
        return _emit_reduction_shared_thread(
            dst_br, src_br, accum, op_type, sctx, reduce_dims, spatial_dims
        )
    else:
        fail(f"unsupported exec_scope {sctx.scope_kind} for reduction_shared_impl")


# ---------------------------------------------------------------------------
# Registration: shared memory reduction (priority=10)
# ---------------------------------------------------------------------------

for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:

    @register_dispatch(
        op_name,
        "cuda",
        variant="shared",
        priority=10,
        when=[
            predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["shared*"]),
            predicate("shared_valid", validate_reduction_shared),
        ],
    )
    def _shared_dispatch(
        op: TilePrimitiveCall, sctx: DispatchContext, _op_type=op_type
    ) -> PrimFunc:
        op = TilePrimitiveCall.downcast(op)
        return reduction_shared_impl(op, _op_type, sctx)
