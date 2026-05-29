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

"""CUDA copy dispatch: scalar ld/st loop (fallback).

Registered ops: copy (variant=default, priority=0).
"""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..exec_scope_utils import exec_scope_ok
from .utils import _is_valid_copy, copy_default_impl


# === Variant: copy/default (priority=0) ===
#
# When: any valid copy op where vec_load predicates fail (e.g. non-power-of-2
# extent, or unsupported scope pair for vectorization). Scalar element loop.
#
# After: nested for-loops over each dimension, one element at a time:
#     for i in Tx.serial(ext0):
#         for j in Tx.serial(ext1):
#             dst[dst_st0+i, dst_st1+j] = src[src_st0+i, src_st1+j]
@register_dispatch(
    "copy",
    "cuda",
    variant="default",
    priority=0,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["cta", "thread"]),
    ],
)
def copy_schedule_default(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    # Conservative scalar fallback
    return copy_default_impl(op_call, sctx)
