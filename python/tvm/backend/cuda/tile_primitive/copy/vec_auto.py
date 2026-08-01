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

"""Auto-vectorizing synchronous copy dispatch."""

from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from .vec_auto_gmem_smem import _emit_gmem_smem, _is_gmem_smem
from .vec_auto_reg import _emit_reg, _is_reg_copy


def _is_vec_auto_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    g_ok, g_reason = _is_gmem_smem(op_call, sctx)
    if g_ok:
        return True, None
    r_ok, r_reason = _is_reg_copy(op_call, sctx)
    if r_ok:
        return True, None
    return False, f"gmem_smem: {g_reason}; reg: {r_reason}"


@register_dispatch(
    "copy",
    "cuda",
    variant="vec_auto",
    priority=10,
    when=[predicate("vec_auto_applicable", _is_vec_auto_copy)],
)
def copy_schedule_vec_auto(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    g_ok, g_reason = _is_gmem_smem(op_call, sctx)
    if g_ok:
        return _emit_gmem_smem(op_call, sctx)
    r_ok, r_reason = _is_reg_copy(op_call, sctx)
    if r_ok:
        return _emit_reg(op_call, sctx)
    fail(f"gmem_smem: {g_reason}; reg: {r_reason}")
