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
"""A rule for DecodeGEMV."""
from typing import List, Optional, Set, Tuple, Union

from tvm import arith, tir
from tvm.ir import structural_equal
from tvm.target import Target

from ..base import (
    BlockInfo,
    ScheduleRule,
    normalize_prim_func,
    try_inline_contiguous_spatial,
)
from . import utils


def _get_reduction_expr(block: tir.Block) -> Optional[tir.PrimExpr]:
    # Detect and return `Y` in `X[...] = X[...] + Y`
    buffer_store = block.body
    if not isinstance(buffer_store, tir.BufferStore):
        return None
    if not isinstance(buffer_store.value, tir.Add):
        return None
    if not structural_equal(
        buffer_store.value.a,
        tir.BufferLoad(buffer_store.buffer, block.body.indices),
        map_free_vars=True,
    ):
        return None
    return buffer_store.value.b


def _detect_dominant_read(block: tir.Block) -> tir.PrimExpr:
    dominant_read, read_iters = None, None
    tir_vars: Set[tir.Var] = set()
    for buffer_region in block.reads:
        tir_vars.clear()

        def _collect_tir_var(expr):
            if isinstance(expr, tir.Var):
                tir_vars.add(expr)

        for expr in buffer_region.region:
            assert expr.extent == 1
            tir.stmt_functor.post_order_visit(expr.min, _collect_tir_var)

        if read_iters is None or read_iters < len(tir_vars):
            read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    (result,) = dominant_read.buffer.offset_of([e.min for e in dominant_read.region])
    return result


class DecodeGEMV(ScheduleRule):
    """A rule for DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc):
            return None
        sch = tir.Schedule(func)
        block_infos = try_inline_contiguous_spatial(sch, normalize_prim_func(sch))
        if block_infos is None or len(block_infos) > 2:
            return None

        block_info = block_infos[0]
        block = block_info.block_rv
        block_stmt = sch.get(block)

        # Step 1. Check reduction block
        if (
            (not block_info.is_reduction())
            or len(block_stmt.writes) != 1
            or _get_reduction_expr(block_stmt) is None
        ):
            return None
        # Step 2. Normalize the block, merge spatial and reduction iters
        is_inner_reduction, c_factor = self._normalize(
            sch,
            block_info,
            arith.normalize_to_iter_sum(
                _detect_dominant_read(block_stmt),
                input_iters={i.var: i.dom for i in block_stmt.iter_vars},
            ),
        )
        if is_inner_reduction is None and c_factor is None:
            return None
        # Step 3. Do the scheduling
        if is_inner_reduction:
            self._sch_inner_reduction(sch, target, block, c_factor)
        else:
            self._sch_inner_spatial(sch, target, block, c_factor)
        # Step 4. Schedule epilogue
        if len(block_infos) == 2:
            sch.set_scope(block, 0, "local")
            sch.reverse_compute_at(block_infos[1].block_rv, sch.get_loops(block)[0])
        return sch

    def _normalize(
        self,
        sch: tir.Schedule,
        block_info: BlockInfo,
        iter_sum: arith.IterSumExpr,
    ) -> Tuple[Optional[bool], Optional[int]]:
        if iter_sum.base != 0:
            return None, None
        iter_to_info = {i.var: i for i in block_info.iters}
        s_dom, r_dom, c_dom, c_factor = None, None, None, None
        for split in iter_sum.args:
            var = split.source.source
            info = iter_to_info[var]
            dom = info.dom
            is_inner_reduction = info.kind == "R"
            if split.lower_factor > 1:
                if c_dom is not None:
                    return None, None
                c_dom = tir.floormod(var, split.lower_factor)
                var = tir.floordiv(var, split.lower_factor)
                dom = tir.floordiv(dom, split.lower_factor)
                if not is_inner_reduction:
                    c_factor = split.lower_factor
            if is_inner_reduction:
                if r_dom is None:
                    r_dom = var
                else:
                    r_dom = r_dom * dom + var
            else:
                if s_dom is None:
                    s_dom = var
                else:
                    s_dom = s_dom * dom + var

        assert r_dom is not None
        if s_dom is None:
            s_dom = tir.const(1, r_dom.dtype)
        if c_dom is None:
            c_dom = tir.const(1, r_dom.dtype)
        sch.transform_block_layout(
            block_info.block_rv,
            tir.IndexMap(
                [i.var for i in block_info.iters],
                [s_dom, r_dom, c_dom],
                None,
            ),
        )
        return is_inner_reduction, c_factor

    def _sch_inner_reduction(
        self,
        sch: tir.Schedule,
        target: Target,
        block: tir.schedule.BlockRV,
        unroll_spatial_factor: Optional[int],
    ):
        # pylint: disable=invalid-name
        _, r, _ = sch.get_loops(block)
        (len_tx,) = utils.suggest_threads_per_block(  # pylint: disable=unbalanced-tuple-unpacking
            target, [sch.get(r)]
        )

        _, tx = sch.split(r, factors=[None, len_tx])
        # Schedule the RF block
        rf = sch.rfactor(tx, 0)
        bx, r, tx, _ = sch.get_loops(rf)
        sch.reorder(bx, tx, r)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, r)
        # Schedule the write back block
        sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
        _, tx, *s = sch.get_loops(block)
        s = sch.fuse(*s)
        sch.reorder(s, tx)
        if unroll_spatial_factor:
            s, inner = sch.split(s, factors=[None, unroll_spatial_factor])
            sch.reorder(s, tx, inner)
        sch.bind(tx, "threadIdx.x")
        # pylint: enable=invalid-name

    def _sch_inner_spatial(
        self,
        sch: tir.Schedule,
        _: Target,
        block: tir.schedule.BlockRV,
        unroll_spatial_factor: Optional[int],
    ):
        # pylint: disable=invalid-name
        s, r, _ = sch.get_loops(block)
        len_tx, len_ty = 16, 16
        _, _ = sch.split(s, factors=[None, len_tx])
        _, ty = sch.split(r, factors=[None, len_ty])
        # Schedule the RF block
        rf = sch.rfactor(ty, 0)
        bx, tx, r, ty, _ = sch.get_loops(rf)
        sch.reorder(bx, tx, ty, r)
        sch.bind(tx, "threadIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, r)
        # Schedule the write back block
        sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
        _, r, *s = sch.get_loops(block)
        s = sch.fuse(*s)
        sch.reorder(s, r)
        if unroll_spatial_factor:
            s, inner = sch.split(s, factors=[None, unroll_spatial_factor])
            sch.reorder(s, r, inner)
        sch.bind(s, "threadIdx.x")
        sch.bind(r, "threadIdx.y")
        # pylint: enable=invalid-name
