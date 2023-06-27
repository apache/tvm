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

from typing import List, Union
from functools import reduce
import tvm
from tvm import tir
from tvm.tir import Schedule
from tvm._ffi import get_global_func
from tvm.target import Target
from tvm.tir.schedule import BlockRV

from ..base import BlockInfo, ScheduleRule, try_inline


class DecodeGEMV(ScheduleRule):
    def __init__(self) -> None:
        super().__init__()
        self.is_trivial_binding = get_global_func("tir.schedule.IsTrivialBinding")
        self.is_reduction = get_global_func("tir.schedule.IsReductionBlock")
        self.get_block_realize = get_global_func("tir.schedule.GetBlockRealize")
        self.get_loop_iter_type = get_global_func("tir.schedule.GetLoopIterType")

    def detect_gemv(self, sch: Schedule, scope_block_rv: BlockRV, block_rv: BlockRV):
        if not self.is_trivial_binding(sch, block_rv):
            return None
        if not self.is_reduction(sch, block_rv, scope_block_rv):
            return None

        block: tir.Block = sch.get(block_rv)
        block_realize: tir.BlockRealize = self.get_block_realize(sch, block_rv)
        block_iters = block.iter_vars
        loops = sch.get_loops(block_rv)
        var_loop_map = {sch.get(loop).loop_var: loop for loop in loops}
        var_range_map = {iter.var: iter.dom for iter in block_iters}
        bv_loop_map, loop_iter_type_map = dict(), dict()
        for bv, binding in zip(block_iters, block_realize.iter_values):
            bv_loop_map[bv.var] = var_loop_map[binding]
            loop_iter_type_map[var_loop_map[binding]] = bv.iter_type

        # C[S0] = C[S0] + f(A_i[S_i, R]), S_i >= S_{i+1}
        # reduce to (appromximately if we ignore smaller buffer accesses)
        # C[S0] = C[S0] + A_0[S_0, R], which is just a reduction
        # further simplification: The order of S0 in C and A_0 are the same
        if not isinstance(block.body, tir.BufferStore) or not isinstance(block.body.value, tir.Add):
            return None

        lhs = block.body.value.a
        rhs = block.body.value.b
        if not isinstance(lhs, tir.BufferLoad):
            lhs, rhs = rhs, lhs
        if not isinstance(lhs, tir.BufferLoad):
            return None

        # TODO: consider visit the body to collect buffer access
        reads = sorted(
            block.reads,
            key=lambda read: reduce(lambda x, y: x * y, read.buffer.shape),
            reverse=True,
        )

        # reads[0] is the buffer that decides the iteration space
        access = list()
        for r_range in reads[0].region:
            if r_range.extent != 1:
                return None
            access.append(r_range.min)
        index = reads[0].buffer.offset_of(access)
        assert len(index) == 1
        index = index[0]
        res = tvm.arith.normalize_to_iter_sum(index, var_range_map)
        assert isinstance(res, tvm.arith.IterSumExpr)
        if res.base != 0:
            return None

        # lhs and rhs use the same set of spatial variables
        lhs_vars = set()
        for value in lhs.indices:
            if not (var_range_map[value].extent == 1 and var_range_map[value].min == 0):
                lhs_vars.add(value)

        # allow at most 1 iter to have lower factor > 1
        S_order, R_order = list(), list()
        loop_c = None
        for split in res.args:
            bv = split.source.source
            if bv in lhs_vars:
                S_order.append(bv_loop_map[bv])
            else:
                R_order.append(bv_loop_map[bv])
            if split.lower_factor > 1:
                if loop_c is not None:
                    return None
                loop_c = bv_loop_map[bv], split.lower_factor

        if len(lhs_vars) != len(S_order):
            return None

        # handle unit loops
        for loop_rv in loops:
            loop: tir.For = sch.get(loop_rv)
            if loop_rv not in S_order and loop_rv not in R_order:
                assert loop.min == 0 and loop.extent == 1
                if loop_iter_type_map[loop_rv] == tir.IterVar.DataPar:
                    S_order.append(loop_rv)
                elif loop_iter_type_map[loop_rv] == tir.IterVar.CommReduce:
                    R_order.append(loop_rv)
                else:
                    raise RuntimeError("Unknown loop type")

        return (
            S_order,
            R_order,
            "S" if res.args[-1].source.source in lhs_vars else "R",
            loop_c,
        )

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if target.kind.name == "cuda":
            len_tx, len_ty = 16, 16
        else:
            len_tx, len_ty = 8, 8

        def _inline_all_spatial():
            blocks = []
            spatial_blocks = []
            for block in sch.get_child_blocks(sch.get_block("root")):
                block = BlockInfo(sch, block)
                if block.is_spatial():
                    spatial_blocks.append(block)
                elif spatial_blocks:
                    blocks.extend(try_inline(sch, spatial_blocks))
                    blocks.append(block)
                    spatial_blocks = []
                else:
                    blocks.append(block)
            if spatial_blocks:
                blocks.extend(try_inline(sch, spatial_blocks))
            return blocks

        sch = tir.Schedule(func)
        blocks = _inline_all_spatial()
        assert len(blocks) <= 2

        pattern = self.detect_gemv(sch, sch.get_block("root"), blocks[0].block)
        if pattern is None:
            print("Mismatch")
            return None

        S_order, R_order, inner_most, loop_c = pattern

        # split the compressed dim out, and reorder the loops according to the pattern
        loop_c_in_S = False
        if loop_c is not None:
            loop_c, factor = loop_c
            outer, inner = sch.split(loop_c, factors=[None, factor])
            if loop_c in S_order:
                loop_c_in_S = True
                S_order[S_order.index(loop_c)] = outer
            else:
                R_order[R_order.index(loop_c)] = outer
            sch.reorder(*(S_order + R_order + [inner]))
        else:
            sch.reorder(*(S_order + R_order))
        # fuse the loops, the loop structure afterwards is [S, R, [inner]]
        S = sch.fuse(*S_order)
        R = sch.fuse(*R_order)
        if inner_most == "S":
            bx, tx = sch.split(S, factors=[None, len_tx], preserve_unit_iters=True)
            R, ty = sch.split(R, factors=[None, len_ty], preserve_unit_iters=True)
            rf = sch.rfactor(ty, 0)

            bx, tx, R, ty = sch.get_loops(rf)[:4]
            sch.reorder(bx, tx, ty, R)
            sch.reverse_compute_at(blocks[0].block, bx, preserve_unit_loops=True)
            sch.bind(tx, "threadIdx.x")
            sch.bind(ty, "threadIdx.y")
            sch.bind(bx, "blockIdx.x")
        else:
            R, tx = sch.split(R, factors=[None, len_tx * len_ty], preserve_unit_iters=True)
            rf = sch.rfactor(tx, 0)

            S, R, tx = sch.get_loops(rf)[:3]
            sch.reorder(S, tx, R)
            sch.reverse_compute_at(blocks[0].block, S, preserve_unit_loops=True)
            sch.bind(tx, "threadIdx.x")
            sch.bind(S, "blockIdx.x")

        # bind the cross thread reduce block
        S_ctr_order, R_ctr_order = list(), list()
        for loop_rv in sch.get_loops(blocks[0].block)[1:]:
            iter_type = self.get_loop_iter_type(sch, loop_rv)
            if iter_type == "S":
                S_ctr_order.append(loop_rv)
            elif iter_type == "R":
                R_ctr_order.append(loop_rv)
            else:
                raise RuntimeError("Unknown loop type " + str(iter_type))
        sch.reorder(*(S_ctr_order + R_ctr_order))
        S_ctr = sch.fuse(*S_ctr_order)
        R_ctr = sch.fuse(*R_ctr_order)

        if loop_c_in_S:
            S_ctr, inner = sch.split(S_ctr, factors=[None, factor], preserve_unit_iters=True)
            sch.reorder(S_ctr, R_ctr, inner)

        if inner_most == "S":
            sch.bind(S_ctr, "threadIdx.x")
            sch.bind(R_ctr, "threadIdx.y")
        else:
            sch.bind(R_ctr, "threadIdx.x")

        sch.set_scope(rf, 0, "local")
        sch.decompose_reduction(rf, sch.get_loops(rf)[2 if inner_most == "R" else 3])

        if len(blocks) == 2:
            sch.set_scope(blocks[0].block, 0, "local")
            sch.reverse_compute_at(blocks[1].block, sch.get_loops(blocks[0].block)[0])

        return sch
