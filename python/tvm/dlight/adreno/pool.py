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
""" Pool schedule rule for Adreno operators."""

from tvm import tir
from tvm.target import Target

from .base import AdrenoScheduleRule
from ..base import analysis


# pylint: disable=invalid-name, unused-variable
class Pool2D(AdrenoScheduleRule):
    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> tir.Schedule:
        sch = tir.Schedule(func)
        root = sch.get_block(name="root", func_name="main")

        blocks = sch.get_child_blocks(root)
        blocks_names = [sch.get(blk).name_hint for blk in blocks]

        if not "adaptive_pool_sum" in blocks_names and not "pool_max" in blocks_names:
            return None

        def schedule_pad(blk: tir.schedule.BlockRV):
            lps, veclp = sch.get_loops(blk)[:-1], sch.get_loops(blk)[-1]
            sch.vectorize(veclp)
            b = sch.fuse(*lps)
            tx_extent = min(int(sch.get(b).extent) & ~int(sch.get(b).extent - 1), 256)
            bx, tx = sch.split(b, [None, tx_extent])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

        def schedule_max_pool(blk: tir.schedule.BlockRV):
            block_info = analysis.get_block_info(sch, blk)
            iters_kind = "".join([_iter.kind for _iter in block_info.iters])
            if iters_kind != "SSSSSRR":
                return None

            lps = sch.get_loops(blk)
            block_lps, vec_lp, red_lps = lps[:4], lps[4], lps[5:]
            write_blk = sch.cache_write(blk, 0, "local")
            sch.reverse_compute_at(write_blk, vec_lp)
            b = sch.fuse(*block_lps)
            tx_extent = min(int(sch.get(b).extent) & ~int(sch.get(b).extent - 1), 256)
            bx, tx = sch.split(b, [None, tx_extent])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec_lp)

            return True

        passed_reduction = False
        for blk in blocks:
            if sch.get(blk).name_hint == "pad_temp":
                schedule_pad(blk)
            elif (
                sch.get(blk).name_hint == "adaptive_pool_sum"
                or sch.get(blk).name_hint == "pool_max"
            ):
                ok = schedule_max_pool(blk)
                if not ok:
                    return None
                passed_reduction = True
            else:
                try:
                    if passed_reduction:
                        sch.reverse_compute_inline(blk)
                    else:
                        sch.compute_inline(blk)
                except:  # pylint: disable=bare-except
                    pass
        return sch
