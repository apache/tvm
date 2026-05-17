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

"""CUDA copy dispatch: vectorized ld/st (ld.global.v4, vectorized smem load/store).

Registered ops: copy (variant=vec_load, priority=10).
"""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import CopyInstType, copy_vec_load_impl
from ..exec_scope_utils import exec_scope_ok
from .utils import _is_valid_copy, _scope_allowed, _vec_len_possible


# === Variant: copy/vec_load (priority=10) ===
#
# When: copy between global<->shared, global<->local, or shared<->local, and the
# layout allows vectorized access (vec_len > 1 for the element type).
#
# Before (TilePrimitiveCall):
#     with Tx.cta():
#         Tx.copy(A_smem[0:64, 0:64], A[0:64, 0:64])
#         # A: global float16, A_smem: shared float16
#
# After (thread_cnt=128, vec_len=8):
#     for s in Tx.serial(ceildiv(4096, 8 * 128)):
#         for vec in Tx.vectorized(8):
#             fused = s * 1024 + threadIdx.x * 8 + vec
#             if fused < 4096:
#                 A_smem[fused // 64, fused % 64] = A[fused // 64, fused % 64]
@register_dispatch(
    "copy",
    "cuda",
    variant="vec_load",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("storage_scope", _scope_allowed),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["cta", "thread"]),
        predicate("vec_len", _vec_len_possible),
    ],
)
def copy_schedule_vec_load(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    # Delegate to the fast vectorized path
    return copy_vec_load_impl(op_call, sctx, CopyInstType.NORMAL)
