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
from typing import Callable, List

from tvm import tir
from tvm._ffi import get_global_func
from tvm.target import Target

from ..base import ScheduleRule


def _max_threads_per_block(target: Target) -> int:
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


class Fallback(ScheduleRule):
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
        max_threads_per_block = _max_threads_per_block(target)
        get_loop_iter_type = get_global_func("tir.schedule.GetLoopIterType")

        sch = tir.Schedule(func)
        blocks = sch.get_child_blocks(sch.get_block(sch.mod["main"].body.block.name_hint))

        while True:

            def _try_inline(func: Callable):
                for i, block in enumerate(blocks):
                    try:
                        func(block)
                    except:  # pylint: disable=bare-except
                        continue
                    return i
                return None

            i = _try_inline(sch.compute_inline)
            if i is None:
                i = _try_inline(sch.reverse_compute_inline)
            if i is None:
                break
            blocks.pop(i)

        for block in blocks:
            s_loops: List[tir.schedule.LoopRV] = []
            r_loops: List[tir.schedule.LoopRV] = []
            o_loops: List[tir.schedule.LoopRV] = []
            for loop in sch.get_loops(block):
                iter_type = get_loop_iter_type(sch, loop)
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
        return sch
