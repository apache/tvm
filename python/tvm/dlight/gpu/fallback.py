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
from typing import List, Tuple

from tvm import tir
from tvm.target import Target

from ..base import normalize_prim_func, try_inline
from . import utils
from .base import GPUScheduleRule


class Fallback(GPUScheduleRule):
    """
    A fallback schedule rule for all GPU operators. It will try to inline all the blocks first,
    and then apply a simple block/grid mapping to the spatial loops on top of the remaining blocks.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> tir.Schedule:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        max_threads_per_block = utils.max_threads_per_block(target)

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)

        if block_infos is None:
            return None

        block_infos = try_inline(sch, block_infos)
        reduction_blocks: List[Tuple[tir.schedule.BlockRV, tir.schedule.LoopRV]] = []
        for block in block_infos:
            s_loops: List[tir.schedule.LoopRV] = []
            r_loops: List[tir.schedule.LoopRV] = []
            o_loops: List[tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block = block.block_rv

            if (
                any(
                    [
                        sch.get(loop_rv).thread_binding is not None
                        for loop_rv in sch.get_loops(block)
                    ]
                )
                or len(sch.get_loops(block)) == 0
            ):
                continue

            for loop, iter_type in zip(sch.get_loops(block), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block))
            sch.reorder(*s_loops, *r_loops, *o_loops)
            bx, tx = sch.split(  # pylint: disable=invalid-name
                sch.fuse(*s_loops),
                factors=[None, max_threads_per_block],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            if len(r_loops) > 0:
                reduction_blocks.append((block, r_loops[0]))

        for block, r_loop in reduction_blocks:
            sch.decompose_reduction(block, r_loop)

        return sch
