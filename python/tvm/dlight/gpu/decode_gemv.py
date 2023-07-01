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
# pylint: disable=missing-docstring
"""A fallback schedule rule for GPU operators."""
# pylint: disable=invalid-name

from typing import List, Optional, Union

from tvm import tir
from tvm._ffi import get_global_func
from tvm.arith import normalize_to_iter_sum
from tvm.ir import structural_equal
from tvm.target import Target

from ..base import ScheduleRule, normalize_prim_func, try_inline_contiguous_spatial


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
    tir_vars = set()
    for buffer_region in block.reads:
        tir_vars.clear()

        def _collect_tir_var(e):
            if isinstance(e, tir.Var):
                tir_vars.add(e)

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
    def __init__(self) -> None:
        super().__init__()
        self.get_loop_iter_type = get_global_func("tir.schedule.GetLoopIterType")

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc):
            return None

        if target.kind.name == "cuda":
            len_tx, len_ty = 16, 16
        else:
            len_tx, len_ty = 8, 8

        sch = tir.Schedule(func)
        block_infos = try_inline_contiguous_spatial(sch, normalize_prim_func(sch))

        if block_infos is None or len(block_infos) > 2:
            return None

        block_info = block_infos[0]
        block = block_info.block_rv
        block_stmt = sch.get(block)

        # Step 1. Check reduction block
        if not block_info.is_reduction():
            return None
        if len(block_stmt.writes) != 1:
            return None
        if _get_reduction_expr(block_stmt) is None:
            return None

        # Step 2. Sort out the spatial and reduction loops
        sorted_iter_access = normalize_to_iter_sum(
            _detect_dominant_read(block_stmt),
            input_iters={i.var: i.dom for i in block_stmt.iter_vars},
        )
        if sorted_iter_access.base != 0:
            return None
        iter_to_info = {i.var: i for i in block_info.iters}
        s_loops, r_loops, c_loops = [], [], []
        for split in sorted_iter_access.args:
            block_var = split.source.source
            block_var_info = iter_to_info[block_var]
            loop_rv = block_var_info.loop_rv
            is_inner_reduction = block_var_info.kind == "R"
            if split.lower_factor > 1:
                c_loop_factor = split.lower_factor
                loop_rv, c_loop = sch.split(loop_rv, factors=[None, c_loop_factor])
                c_loops.append(c_loop)
                is_loop_c_reduction = is_inner_reduction
            if is_inner_reduction:
                r_loops.append(loop_rv)
            else:
                s_loops.append(loop_rv)

        if len(c_loops) > 1:
            return None
        if len(s_loops) != len([_ for i in block_info.iters if i.kind == "S"]):
            return None
        if len(s_loops) == 0 or len(r_loops) == 0:
            return None

        sch.reorder(*s_loops, *r_loops, *c_loops)
        s = sch.fuse(*s_loops)
        r = sch.fuse(*r_loops)

        if is_inner_reduction:
            _, tx = sch.split(r, factors=[None, len_tx * len_ty])
            rf = sch.rfactor(tx, 0)
            s, r, tx = sch.get_loops(rf)[:3]
            sch.reorder(s, tx, r)
            sch.reverse_compute_at(block, s, preserve_unit_loops=True)
            sch.bind(tx, "threadIdx.x")
            sch.bind(s, "blockIdx.x")
        else:
            sch.split(s, factors=[None, len_tx])
            _, ty = sch.split(r, factors=[None, len_ty])
            rf = sch.rfactor(ty, 0)
            bx, tx, r, ty = sch.get_loops(rf)[:4]
            sch.reorder(bx, tx, ty, r)
            sch.reverse_compute_at(block, bx, preserve_unit_loops=True)
            sch.bind(tx, "threadIdx.x")
            sch.bind(ty, "threadIdx.y")
            sch.bind(bx, "blockIdx.x")

        s_loops, r_loops = [], []
        for loop_rv in sch.get_loops(block)[1:]:
            iter_type = self.get_loop_iter_type(sch, loop_rv)
            if iter_type == "S":
                s_loops.append(loop_rv)
            elif iter_type == "R":
                r_loops.append(loop_rv)
            else:
                raise RuntimeError("Unknown loop type " + str(iter_type))
        sch.reorder(*s_loops, *r_loops)
        s_ctr = sch.fuse(*s_loops)
        r_ctr = sch.fuse(*r_loops)

        if c_loops and not is_loop_c_reduction:
            s_ctr, inner = sch.split(s_ctr, factors=[None, c_loop_factor])
            sch.reorder(s_ctr, r_ctr, inner)

        if is_inner_reduction:
            sch.bind(r_ctr, "threadIdx.x")
            sch.set_scope(rf, 0, "local")
            sch.decompose_reduction(rf, sch.get_loops(rf)[2])
        else:
            sch.bind(s_ctr, "threadIdx.x")
            sch.bind(r_ctr, "threadIdx.y")
            sch.set_scope(rf, 0, "local")
            sch.decompose_reduction(rf, sch.get_loops(rf)[3])

        if len(block_infos) == 2:
            sch.set_scope(block, 0, "local")
            sch.reverse_compute_at(block_infos[1].block_rv, sch.get_loops(block)[0])

        return sch
