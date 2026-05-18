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

"""CUDA reduction operator dispatch: SM100+ packed optimized variant.

Registered ops: sum, max, min.

When: thread scope, all local buffers, float32, 1D src with len >= 8,
SM100+ (uses packed PTX instructions not available on older GPUs).

Before (TilePrimitiveCall -- sum example):
    with Tx.thread():
        Tx.sum(dst_local[0:1], src_local[0:32])   # float32, reduce 32 -> 1

After -- packed_add_sum (uses add.f32x2 to reduce pairs):
    with Tx.thread():
        # Iteratively reduce: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Each step: add.f32x2 combines adjacent pairs
        for i in Tx.serial(16):
            Tx.cuda.func_call("add_f32x2", &buf[i*2], &buf[i*2], &buf[i*2+2])
        # ... repeat halving until scalar result
        dst_local[0] = buf[0]

After -- 3input_maxmin (uses 3-input PTX max/min):
    with Tx.thread():
        # Tree reduction with 3-input instructions:
        # max(a, b, c) in one PTX instruction
        for i in Tx.serial(n // 3):
            Tx.cuda.func_call("max3_f32", &buf[i*3], &buf[i*3+1], &buf[i*3+2])

With accum=True: accumulator folded into first element/pair of the reduction.
"""

import functools
import operator

from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ...common import ReduceOpType
from ..common import sm_version_ok
from ..exec_scope_utils import exec_scope_ok
from .utils import (
    _dst_len_ok,
    _dtype_ok,
    _local_scope_match,
    _reduction_len_ok,
    _src_ndim_ok,
    reduce_op_table,
)


def _emit_reduction_local_thread_packed_add_sum(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc:
    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    [r.extent for r in dst_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def impl():
        with Tx.thread():
            local_sum = Tx.alloc_buffer([8], dtype, scope="local")
            # First pass: copy first 8 elements (with optional accumulator)
            for i in Tx.unroll(8):
                if accum and i == 0:
                    local_sum[i] = src[src_base + i] + dst[tuple(dst_st)]
                else:
                    local_sum[i] = src[src_base + i]

            # Process remaining full chunks of 8
            for outer in Tx.serial(num_full_chunks - 1):
                for j in Tx.unroll(4):
                    Tx.ptx.add_f32x2(
                        Tx.address_of(local_sum[2 * j]),
                        Tx.cuda.make_float2(local_sum[2 * j], local_sum[2 * j + 1]),
                        Tx.cuda.make_float2(
                            src[src_base + 8 * (outer + 1) + 2 * j],
                            src[src_base + 8 * (outer + 1) + 2 * j + 1],
                        ),
                        ftz=True,
                    )

            # Handle remainder elements (0 to 7)
            for i in Tx.serial(remainder):
                local_sum[0] = local_sum[0] + src[src_base + remainder_base + i]

            # Final packed add sum: 8 -> 4 -> 2 -> 1
            Tx.ptx.add_f32x2(
                Tx.address_of(local_sum[0]),
                Tx.cuda.make_float2(local_sum[0], local_sum[1]),
                Tx.cuda.make_float2(local_sum[2], local_sum[3]),
                ftz=True,
            )
            Tx.ptx.add_f32x2(
                Tx.address_of(local_sum[4]),
                Tx.cuda.make_float2(local_sum[4], local_sum[5]),
                Tx.cuda.make_float2(local_sum[6], local_sum[7]),
                ftz=True,
            )
            Tx.ptx.add_f32x2(
                Tx.address_of(local_sum[0]),
                Tx.cuda.make_float2(local_sum[0], local_sum[1]),
                Tx.cuda.make_float2(local_sum[4], local_sum[5]),
                ftz=True,
            )
            dst[tuple(dst_st)] = local_sum[0] + local_sum[1]
    # fmt: on

    return impl


def _emit_reduction_local_thread_3input_maxmin(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    accum: bool,
    reduce_op: ReduceOpType,
    sctx: DispatchContext,
) -> PrimFunc:
    dst, src = dst_buffer_region.buffer, src_buffer_region.buffer
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    dtype = src.dtype

    src_extent = [r.extent for r in src_region]
    src_st = [r.min for r in src_region]
    dst_st = [r.min for r in dst_region]

    reduction_len = functools.reduce(operator.mul, src_extent, 1)

    op_func = reduce_op_table[reduce_op]
    reduce3_func = (
        Tx.ptx.reduce3_max_f32 if reduce_op == ReduceOpType.MAX else Tx.ptx.reduce3_min_f32
    )

    src_base = src_st[0]
    num_full_chunks = reduction_len // 8
    remainder = reduction_len % 8
    remainder_base = num_full_chunks * 8

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def impl():
        with Tx.thread():
            temp = Tx.alloc_buffer([4], dtype, scope="local")
            # First pass: process first 8 elements into 4 temps
            for i in Tx.unroll(4):
                if accum and i == 0:
                    temp[i] = reduce3_func(src[src_base + 2 * i], src[src_base + 2 * i + 1], dst[tuple(dst_st)])  # noqa: E501
                else:
                    temp[i] = op_func(src[src_base + 2 * i], src[src_base + 2 * i + 1])

            # Process remaining full chunks of 8
            for outer in Tx.serial(num_full_chunks - 1):
                for i in Tx.unroll(4):
                    temp[i] = reduce3_func(
                        temp[i],
                        src[src_base + 8 * (outer + 1) + 2 * i],
                        src[src_base + 8 * (outer + 1) + 2 * i + 1],
                    )

            # Process remainder elements (0 to 7 elements)
            for i in Tx.serial(remainder):
                temp[0] = op_func(temp[0], src[src_base + remainder_base + i])

            # Final merge: combine 4 temps into result
            dst[tuple(dst_st)] = op_func(temp[0], temp[1])
            dst[tuple(dst_st)] = reduce3_func(dst[tuple(dst_st)], temp[2], temp[3])
    # fmt: on

    return impl


def _sm100_packed_add_sum_impl(op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext):
    op = TilePrimitiveCall.downcast(op)
    return _emit_reduction_local_thread_packed_add_sum(op.output, op.input, op.accum, op_type, sctx)


def _sm100_3input_maxmin_impl(op: TilePrimitiveCall, op_type: ReduceOpType, sctx: DispatchContext):
    op = TilePrimitiveCall.downcast(op)
    return _emit_reduction_local_thread_3input_maxmin(op.output, op.input, op.accum, op_type, sctx)


_optimized_local_reduction_predicates = [
    predicate("exec_scope", exec_scope_ok, expected_scopes=["thread"]),
    predicate("local_scope", _local_scope_match),
    predicate("dst_len", _dst_len_ok, expected_len=1),
    predicate("src_ndim", _src_ndim_ok, expected_ndim=1),
    predicate("dtype", _dtype_ok, expected_dtype="float32"),
    predicate("sm_version", sm_version_ok, min_version=100),
    predicate("reduction_len", _reduction_len_ok, min_len=8),
]

_optimized_impl_table = {
    ReduceOpType.SUM: ("packed_add_sum", _sm100_packed_add_sum_impl),
    ReduceOpType.MAX: ("3input_maxmin", _sm100_3input_maxmin_impl),
    ReduceOpType.MIN: ("3input_maxmin", _sm100_3input_maxmin_impl),
}


# ---------------------------------------------------------------------------
# Registration: SM100+ optimized local reduction (priority=20)
# ---------------------------------------------------------------------------

for op_name, op_type in [
    ("sum", ReduceOpType.SUM),
    ("max", ReduceOpType.MAX),
    ("min", ReduceOpType.MIN),
]:
    variant_name, optimized_impl = _optimized_impl_table[op_type]

    @register_dispatch(
        op_name,
        "cuda",
        variant=variant_name,
        priority=20,
        when=_optimized_local_reduction_predicates,
    )
    def _optimized_dispatch(
        op: TilePrimitiveCall, sctx: DispatchContext, _impl=optimized_impl, _op_type=op_type
    ) -> PrimFunc:
        op = TilePrimitiveCall.downcast(op)
        return _impl(op, _op_type, sctx)
