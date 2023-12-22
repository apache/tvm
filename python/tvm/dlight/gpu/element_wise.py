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

from ..base import ScheduleRule, normalize_prim_func, try_inline
from . import utils


class ElementWise(ScheduleRule):
    """
    An elementwise schedule rule for GPU operators.
    """
    def apply_config(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        config,
    ) -> tir.Schedule:
        block_factors = config.block
        thread_factors = config.thread
        step_factors = config.step
        
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)

        if block_infos is None:
            return None

        block_infos = try_inline(sch, block_infos)

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

            block_loops = []
            vthread_loops = []
            thread_loops = []
            inner_loops = []
            for s_loop, block_factor, step_factor, thread_factor in zip(s_loops, block_factors, step_factors, thread_factors):
                block_loop, inner_loop = sch.split(s_loop, factors=[None, block_factor])
                vthread_loop, inner_loop = sch.split(
                inner_loop, factors=[None, thread_factor * step_factor])
                thread_loop, inner_loop = sch.split(inner_loop, factors=[None, step_factor])
                block_loops.append(block_loop)
                vthread_loops.append(vthread_loop)
                thread_loops.append(thread_loop)
                inner_loops.append(inner_loop)
                
            # inner virtual thread first
            vthread_loops = list(reversed(vthread_loops))
            sch.reorder(*block_loops, *vthread_loops, *thread_loops, *inner_loops, *r_loops, *o_loops)
            sch.bind(sch.fuse(*block_loops), "blockIdx.x")
            sch.bind(sch.fuse(*thread_loops), "threadIdx.x")
            for i, ax in enumerate(vthread_loops):
                sch.bind(ax, "vthread" + ['.x', '.y', '.z'][i])

        return sch
