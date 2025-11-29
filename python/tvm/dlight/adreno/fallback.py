# licensed to the apache software foundation (asf) under one
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
"""Dlight Adreno Fallback Schedules"""


from typing import List, Union

from tvm import tir
from tvm.target import Target
from .. import analysis
from .base import AdrenoScheduleRule
from .utils import get_texture_storage


# pylint: disable=invalid-name,missing-function-docstring,unused-variable,unused-import
class Fallback(AdrenoScheduleRule):
    """Texture Based Fallback Schedule(s) for Adreno"""

    @staticmethod
    def schedule_inline_blocks(
        sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]
    ) -> List[tir.schedule.BlockRV]:
        """
        Auto Inlines Injective and Element-wise Operations while trying to omit data pad blocks...
        """

        if blocks is None:
            root_blk = analysis.get_root_block(sch)
            blocks = sch.get_child_blocks(root_blk)

        remaining_blocks = []
        for blk in blocks:
            block_info = analysis.get_block_info(sch, blk)
            if block_info.is_injective() and not block_info.is_data_pad(sch):
                if len(block_info.consumers) == 1:
                    try:
                        sch.compute_inline(blk)
                    except Exception:  # pylint: disable=broad-exception-caught
                        remaining_blocks.append(blk)
                elif len(block_info.producers) == 1:
                    inlined_once = False
                    try:
                        # Would cause an issue inlining to producer with multiple consumers
                        while (
                            len(sch.get_producers(blk)) == 1
                            and len(sch.get_consumers(sch.get_producers(blk)[0])) == 1
                        ):
                            sch.reverse_compute_inline(blk)
                            inlined_once = True
                    except Exception:  # pylint: disable=broad-exception-caught
                        break
                    if not inlined_once:
                        remaining_blocks.append(blk)
                else:
                    remaining_blocks.append(blk)
            else:
                remaining_blocks.append(blk)
        return remaining_blocks

    @staticmethod
    def schedule_default(sch: tir.Schedule, blk: tir.schedule.BlockRV):
        block_info = analysis.get_block_info(sch, blk)

        s_loops, r_loops, o_loops = [], [], []
        v_loop = block_info.write_bufs[0].assoc_lps[-1]

        for iter_info in block_info.iters:
            if sch.get(iter_info.loop_rv) == sch.get(v_loop):
                continue
            {"S": s_loops, "R": r_loops, "O": o_loops}.get(iter_info.kind).append(iter_info.loop_rv)

        iter_vars = analysis.collect_block_iter_vars_used_in_access_region(
            block_info.block_stmt, block_info.write_bufs[0].buf_region.region
        )
        o_outer = [lp for lp in o_loops if sch.get(lp).var in iter_vars]
        o_inner = [lp for lp in o_loops if sch.get(lp).var not in iter_vars]

        # Can't change loop order for opaque loops
        if o_loops != o_outer + o_inner:
            return

        o_outer.append(v_loop)
        sch.reorder(*s_loops, *o_outer, *r_loops, *o_inner)

        assert s_loops
        tgt = Target.current(allow_none=True)

        b = sch.fuse(*s_loops)
        tx_extent = analysis.get_max_threads_per_block(tgt) if tgt is not None else 256
        bx, tx = sch.split(b, [None, tx_extent])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        if len(r_loops) > 1:
            lp = [*s_loops, *o_outer][-1]
            init_block = sch.decompose_reduction(blk, lp)
            wblk = sch.cache_write(blk, 0, "local")
            sch.compute_at(wblk, lp)
            if v_loop:
                sch.vectorize(sch.get_loops(init_block)[-1])
                sch.vectorize(sch.get_loops(wblk)[-1])
        elif v_loop is not None:
            sch.vectorize(v_loop)

    @staticmethod
    def schedule_fallback(sch):
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        schedule_blocks = [
            blk
            for blk in blocks
            if analysis.get_block_info(sch, blk).is_reduction()
            or analysis.get_block_info(sch, blk).is_data_pad(sch)
        ]
        remaining_blocks = [blk for blk in blocks if blk not in schedule_blocks]

        for blk in schedule_blocks:
            Fallback.schedule_default(sch, blk)
        remaining_blocks = Fallback.schedule_inline_blocks(sch, remaining_blocks)
        # TODO: Analyze unscheduled blocks to schedule instead of relying on remaining
        for blk in remaining_blocks:
            Fallback.schedule_default(sch, blk)

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        # pylint: disable=invalid-name

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if any(len(sch.get_child_blocks(block)) != 0 for block in blocks):
            return None

        block_infos = [analysis.get_block_info(sch, block) for block in blocks]
        if not any("texture" in block.write_bufs[0].get_scope() for block in block_infos):
            return None

        Fallback.schedule_fallback(sch)
        return sch
