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
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""
from typing import List, Union

from tvm import tir
from tvm.target import Target

from ..base import BlockInfo, ScheduleRule, try_inline


class Reduction(ScheduleRule):
    """Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        else:
            len_tx = 64
            unroll_depth = 64

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
        assert len(blocks) > 0

        dom_kind = blocks[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))
        try:
            for block in blocks[1:-1]:
                assert block.dom_kind() == dom_kind
            assert blocks[-1].is_spatial()
            assert len(blocks[-1].dom_kind()) == len(dom_kind)
        except AssertionError:
            print("Mismatch")
            return None

        loops = sch.get_loops(blocks[-1].block)
        bx = sch.fuse(*loops[:num_leading_s])  # pylint: disable=invalid-name
        _, tx = sch.split(loops[-1], [None, len_tx])  # pylint: disable=invalid-name
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for block in reversed(blocks[:-1]):
            block = block.block
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            _, tx = sch.split(r_loop, [None, len_tx])  # pylint: disable=invalid-name
            sch.bind(tx, "threadIdx.x")

        sch.annotate(bx, ann_key="pragma_auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(bx, ann_key="pragma_unroll_explicit", ann_val=1)
        return sch
