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

"""Scalar single-thread copy fallback (priority=0)."""

import warnings

import tvm
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import (
    predicate,
    register_dispatch,
)
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ._common import _TID_AXIS_FOR_SCOPE
from .reg import _axis_decl
from .utils import _is_valid_copy


def _region_st_extent(buffer_region):
    region = buffer_region.region
    return [r.min for r in region], [r.extent for r in region]


def _emit_fallback(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    src_st, src_extent = _region_st_extent(op_call.src)
    dst_st, dst_extent = _region_st_extent(op_call.dst)

    warnings.warn(
        f"copy/fallback (scalar single-thread) picked for {src.scope()} -> "
        f"{dst.scope()} at scope_kind={sctx.scope_kind}; all faster variants "
        f"rejected.",
        stacklevel=2,
    )

    def _copy_body(dst_buf, src_buf):
        dst_indices = [i for i in range(len(dst_buf.shape)) if dst_extent[i] != 1]
        src_indices = [i for i in range(len(src_buf.shape)) if src_extent[i] != 1]
        assert len(dst_indices) == len(src_indices)
        copy_extents = [dst_extent[i] for i in dst_indices]

        def _dst_coord(lvs):
            if isinstance(lvs, tvm.ir.Var):
                lvs = [lvs]
            coord = list(dst_st)
            for k, lv in enumerate(lvs):
                coord[dst_indices[k]] += lv
            return coord

        def _src_coord(lvs):
            if isinstance(lvs, tvm.ir.Var):
                lvs = [lvs]
            coord = list(src_st)
            for k, lv in enumerate(lvs):
                coord[src_indices[k]] += lv
            return coord

        with T.grid(*copy_extents) as lvs:
            T.buffer_store(dst_buf, src_buf[tuple(_src_coord(lvs))], _dst_coord(lvs))

    scope_kind = sctx.scope_kind

    if scope_kind == "thread":

        @T.prim_func(check_well_formed=False)
        def impl():
            _copy_body(dst, src)

        return impl

    tid_axis_name = _TID_AXIS_FOR_SCOPE[scope_kind]
    # first-active tid = composition of per-axis offsets (radix-32, since a warp is 32 lanes)
    first_tid = int(sctx.intra["laneid"][1])
    if scope_kind == "warpgroup":
        first_tid += 32 * int(sctx.intra["wid_in_wg"][1])
    elif scope_kind == "cta":
        first_tid += 32 * int(sctx.intra["warpid"][1])

    @T.prim_func(check_well_formed=False)
    def impl():
        tid = _axis_decl(tid_axis_name, sctx)
        if tid == first_tid:
            _copy_body(dst, src)

    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="fallback",
    priority=0,
    when=[predicate("validate_copy_op", _is_valid_copy)],
)
def copy_schedule_fallback(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit_fallback(op_call, sctx)
