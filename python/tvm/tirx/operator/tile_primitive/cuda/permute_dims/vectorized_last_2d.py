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

"""CUDA permute_dims dispatch: vectorized_permute_dims_last_2d variant."""

import math

from tvm.script import tirx as Tx
from tvm.tirx import Buffer, BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import get_indices, get_st_extent


def validate_deepgemm_permute_dims(op_call: TilePrimitiveCall, sctx: DispatchContext) -> bool:
    op_call = TilePrimitiveCall.downcast(op_call)
    if isinstance(op_call.buffer, Buffer):
        buffer: Buffer = op_call.buffer
        extent = buffer.shape
    elif isinstance(op_call.buffer, BufferRegion):
        buffer: Buffer = op_call.buffer.buffer
        st, extent = get_st_extent(op_call.buffer)

    order = op_call.order
    if sctx.is_warp:
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
        ndim = len(order)
        expected_order = [*list(range(ndim - 2)), ndim - 1, ndim - 2]
        if list(order) != expected_order:
            return False
        if not math.prod(extent[:-2]) == 1:
            return False
        strides = list(buffer.strides)
        if not (strides == [] or (strides[-1] == 1 and strides[-2] == extent[-1])):
            return False
        return True
    return False


def vectorized_permute_dims_last_2d_impl(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> PrimFunc | None:
    op_call = TilePrimitiveCall.downcast(op_call)
    if isinstance(op_call.buffer, Buffer):
        buffer: Buffer = op_call.buffer
        extent = shape = buffer.shape
        st = [0] * len(extent)
    elif isinstance(op_call.buffer, BufferRegion):
        buffer: Buffer = op_call.buffer.buffer
        shape = buffer.shape
        st, extent = get_st_extent(op_call.buffer)

    M, N = extent[-2:]
    vec_len = op_call.config.get("vec_len")

    if vec_len is None:
        for vec_len in range(4, 0, -1):
            if M % vec_len == 0:
                break

    if not shape[-1] % vec_len == 0:
        vec_len = 1
    if not (st[-2] * shape[-1] + st[-1]) % vec_len == 0:
        vec_len = 1

    # Thread and vectorization setup
    if sctx.is_warp:
        tid_x = sctx.launch_params["threadIdx.x"]
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        # fmt: off
        @Tx.prim_func
        def impl():
            warp_size = Tx.meta_var(32)
            lane_id = Tx.meta_var(tid_x % warp_size)
            reg_trans = Tx.alloc_buffer((N // warp_size, M // vec_len, vec_len), buffer.dtype, scope="local")  # noqa: E501
            for wi in Tx.unroll(0, N // warp_size):
                for vi in Tx.unroll(0, M // vec_len):
                    for vec in Tx.unroll(vec_len):
                        old_index = Tx.meta_var(get_indices((vi * vec_len + vec) * N + wi * warp_size + lane_id, st, extent))  # noqa: E501
                        reg_trans[wi, vi, vec] = buffer[tuple(old_index)]
            Tx.cuda.warp_sync()
            for wi in Tx.unroll(0, N // warp_size):
                for vi in Tx.unroll(0, M // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        new_index = Tx.meta_var(get_indices((wi * warp_size + lane_id) * M + vi * vec_len + vec, st, extent))  # noqa: E501
                        buffer[tuple(new_index)] = reg_trans[wi, vi, vec]
            Tx.cuda.warp_sync()
        # fmt: on
    else:
        raise NotImplementedError
    return impl


# === Variant: permute_dims/vectorized_permute_dims_last_2d (priority=20) ===
#
# When: shared-memory buffer with TileLayout, permutation swaps only the last
# 2 dimensions (e.g. [0,1,3,2] for 4D), at warp scope. In-place transpose.
#
# Before (TilePrimitiveCall):
#     with Tx.warp():
#         Tx.permute_dims(A_smem[0:64, 0:64], order=[1, 0])
#         # A_smem: shared float16 (64, 64), in-place transpose
#
# After (warp-level register-buffered transpose, vec_len=4):
#     lane_id = threadIdx.x % 32
#     reg_trans = Tx.alloc_buffer((2, 16, 4), "float16", scope="local")
#     # Phase 1: read rows into registers (each lane reads a column stripe)
#     for wi in Tx.unroll(2):                          # N // warp_size
#         for vi in Tx.unroll(16):                     # M // vec_len
#             for vec in Tx.unroll(4):
#                 reg_trans[wi, vi, vec] = A_smem[(vi*4+vec)*64 + wi*32+lane_id]
#     Tx.cuda.warp_sync()
#     # Phase 2: write back transposed (column index becomes row)
#     for wi in Tx.unroll(2):
#         for vi in Tx.unroll(16):
#             for vec in Tx.vectorized(4):
#                 A_smem[(wi*32+lane_id)*64 + vi*4+vec] = reg_trans[wi, vi, vec]
#     Tx.cuda.warp_sync()
@register_dispatch(
    "permute_dims",
    "cuda",
    variant="vectorized_permute_dims_last_2d",
    priority=20,
    when=[
        predicate(
            "validate_deepgemm_permute_dims",
            lambda op, sctx: (
                validate_deepgemm_permute_dims(op, sctx),
                "validate_deepgemm_permute_dims failed",
            ),
        )
    ],
)
def permute_dims_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    return vectorized_permute_dims_last_2d_impl(op, sctx)
