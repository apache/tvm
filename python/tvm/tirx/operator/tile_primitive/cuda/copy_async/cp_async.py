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

"""copy_async dispatch variant: non-bulk-copy (cp.async)."""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import CopyInstType, copy_vec_load_impl, validate_copy_op


# === Variant: copy_async/non-bulk-copy (priority=20) ===
#
# When: any valid async copy. Highest priority — tried first before TMA.
# Succeeds for global↔shared copies where vectorization works; fails back
# to TMA for single-thread scope or when cp.async doesn't apply.
#
# Before (TilePrimitiveCall):
#     with Tx.cta():
#         Tx.copy_async(A_smem[0:64, 0:64], A[0:64, 0:64])
#
# After (uses cp.async PTX instead of regular load/store):
#     for s in Tx.serial(ceildiv(4096, 8 * 128)):
#         for vec in Tx.vectorized(8):
#             fused = s * 1024 + threadIdx.x * 8 + vec
#             if fused < 4096:
#                 # emitted as cp.async.bulk.shared.global [smem_addr], [gmem_addr], 16
#                 A_smem[idx] = A[idx]
@register_dispatch(
    "copy_async",
    "cuda",
    variant="non-bulk-copy",
    priority=20,
    when=[
        predicate(
            "validate_copy_op", lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op")
        )
    ],
)
def copy_async_dispatch_cp_async(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_vec_load_impl(op, sctx, CopyInstType.CP_ASYNC)
