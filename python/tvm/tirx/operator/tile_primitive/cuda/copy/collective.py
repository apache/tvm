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

"""CUDA copy dispatch for collective per-thread local views."""

import functools
import operator

from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import get_indices, get_st_extent
from ..layout_utils import get_local_region


def _validate_layout_partition(
    layout, buf, st, ext, analyzer: Analyzer
) -> tuple[bool, tuple | None]:
    if layout.is_swizzle():
        return False, None
    if not isinstance(layout, TileLayout):
        return False, None
    if not getattr(layout, "shard", None):
        return False, None
    if not any(it.axis.is_thread() for it in layout.shard):
        return False, None
    for it in layout.shard:
        if it.axis.is_thread() and analyzer.can_prove_equal(it.stride, 0):
            return False, None
    replica = getattr(layout, "replica", None) or []
    if any(it.axis.is_thread() for it in replica):
        return False, None
    local_info = get_local_region(layout, list(buf.shape), st, ext)
    if local_info is None:
        return False, None
    return True, local_info


def _get_distributed_local_info(buf: Buffer, st, ext, analyzer: Analyzer):
    layout = buf.layout
    if buf.scope() != "local" or layout is None or layout.is_trivial():
        return None
    ok, info = _validate_layout_partition(layout, buf, st, ext, analyzer)
    return info if ok else None


def validate_copy_local_view(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_br, src_br = op_call.dst, op_call.src
    dst, src = dst_br.buffer, src_br.buffer

    if not (sctx.is_cuda() and sctx.scope_kind in ["warp", "warpgroup", "cta", "cluster"]):
        return False, f"unsupported exec_scope {sctx.scope_kind}"
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype}, dst={dst.dtype}"

    analyzer = Analyzer()
    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)
    src_local_info = _get_distributed_local_info(src, src_st, src_extent, analyzer)
    dst_local_info = _get_distributed_local_info(dst, dst_st, dst_extent, analyzer)

    if (src_local_info is None) == (dst_local_info is None):
        return False, "expected exactly one side to be thread-distributed local layout"

    if src_local_info is not None:
        _, _, src_local_ext = src_local_info
        src_local_total = functools.reduce(operator.mul, src_local_ext, 1)
        dst_total = functools.reduce(operator.mul, dst_extent, 1)
        if not analyzer.can_prove_equal(src_local_total, dst_total):
            return False, "src per-thread extent mismatch with dst extent"
        return True, None

    assert dst_local_info is not None
    _, _, dst_local_ext = dst_local_info
    dst_local_total = functools.reduce(operator.mul, dst_local_ext, 1)
    src_total = functools.reduce(operator.mul, src_extent, 1)
    if not analyzer.can_prove_equal(dst_local_total, src_total):
        return False, "dst per-thread extent mismatch with src extent"
    return True, None


def copy_local_view_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    del sctx
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_br, src_br = op_call.dst, op_call.src
    dst, src = dst_br.buffer, src_br.buffer

    src_st, src_extent = get_st_extent(src_br)
    dst_st, dst_extent = get_st_extent(dst_br)

    analyzer = Analyzer()
    src_local_info = _get_distributed_local_info(src, src_st, src_extent, analyzer)
    dst_local_info = _get_distributed_local_info(dst, dst_st, dst_extent, analyzer)

    if src_local_info is not None:
        src_local_shape, src_local_st, src_local_ext = src_local_info
        local_total = functools.reduce(operator.mul, src_local_ext, 1)

        # fmt: off
        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                src_local = src.local(*src_local_shape)
                for s in Tx.serial(0, local_total):
                    fused = Tx.meta_var(s)
                    src_idx = Tx.meta_var(get_indices(fused, src_local_st, src_local_ext))
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    dst[tuple(dst_idx)] = src_local[tuple(src_idx)]
        # fmt: on
        return impl

    if dst_local_info is not None:
        dst_local_shape, dst_local_st, dst_local_ext = dst_local_info
        local_total = functools.reduce(operator.mul, dst_local_ext, 1)

        # fmt: off
        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_local = dst.local(*dst_local_shape)
                for s in Tx.serial(0, local_total):
                    fused = Tx.meta_var(s)
                    src_idx = Tx.meta_var(get_indices(fused, src_st, src_extent))
                    dst_idx = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                    dst_local[tuple(dst_idx)] = src[tuple(src_idx)]
        # fmt: on
        return impl

    fail("expected exactly one side to be thread-distributed local layout")


@register_dispatch(
    "copy",
    "cuda",
    variant="local_view",
    priority=15,
    when=[predicate("local_view_valid", validate_copy_local_view)],
)
def copy_schedule_local_view(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_local_view_impl(op_call, sctx)
