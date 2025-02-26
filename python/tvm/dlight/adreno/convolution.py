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
# pylint: disable=missing-docstring, invalid-name
"""A Conv2d schedule rule for Adreno GPU operators."""
from dataclasses import dataclass
from typing import List, Optional

from tvm import tir
from tvm.target import Target
from tvm.tir import IterVar
from tvm.tir.schedule.schedule import BlockRV

from ..analysis import BlockInfo, IterInfo
from .base import AdrenoScheduleRule


def is_spatial_block(sch: tir.Schedule, block: BlockRV) -> bool:
    block_stmt = sch.get(block)
    iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
    return iter_types == {IterVar.DataPar}


def is_reduction_block(sch: tir.Schedule, block: BlockRV) -> bool:
    block_stmt = sch.get(block)
    iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
    return iter_types == {IterVar.CommReduce, IterVar.DataPar}


def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def get_block_info(sch: tir.Schedule, block: tir.schedule.BlockRV) -> BlockInfo:
    def _iter_kind(loop: tir.IterVar) -> str:
        return {tir.IterVar.DataPar: "S", tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

    def _is_reduction_block(block: tir.schedule.BlockRV):
        for iter_var in sch.get(block).iter_vars:
            if _iter_kind(iter_var) == "R":
                return True
        return False

    return BlockInfo(
        name=sch.get(block).name_hint,
        iters=[
            IterInfo(
                kind=_iter_kind(iter_var),
                var=iter_var.var,
                dom=iter_var.dom.extent,
                loop_rv=loop_rv,
            )
            for loop_rv, iter_var in zip(sch.get_loops(block), sch.get(block).iter_vars)
        ],
        block_rv=block,
        reduction_block=_is_reduction_block(block),
    )


def get_reduction_blocks(sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]) -> bool:
    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all(
        [is_reduction_block(sch, block) or is_spatial_block(sch, block) for block in blocks]
    ):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction_block(sch, block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks[0]


def is_convolution(sch: tir.Schedule, block: tir.schedule.BlockRV):
    # TODO: Use buffer access patterns to discover convolution type kernels instead of using name.
    return (
        sch.get(block).name_hint.count("conv2d_NCHWc_OIHWo")
        and "".join([iter_type.kind for iter_type in get_block_info(sch, block).iters])
        == "SSSSSRRR"
    )


class Conv2d(AdrenoScheduleRule):
    """The schedule rule for convolution computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Conv2d.Config(
                block_size_x=8,
                block_size_y=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and (
            ("android" in str(target.host)) or ("adreno" in str(target.attrs))
        ):
            return Conv2d.Config(
                block_size_x=32,
                block_size_y=4,
                vector_size=8,
                unroll=16,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Conv2d.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        if isinstance(func, tir.PrimFunc):
            sch = tir.Schedule(func)

        # config = self.get_configs(target)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        reduction_block = get_reduction_blocks(sch, blocks)

        if reduction_block is None:
            return None
        if not is_convolution(sch, reduction_block):
            return None

        def schedule_data_pad(blk):
            axes = sch.get_loops(blk)
            axes, vec = axes[:-1], axes[-1]
            axis = sch.fuse(*axes)
            bx, ty, tx = sch.split(axis, [None, 16, 16])
            sch.bind(bx, "blockIdx.x")
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec)

        def schedule_conv2d(blk):
            # TODO: Loop Pattern mayn't be reliable, need to perform better analysis.
            n, oc, oh, ow, ob, ic, kh, kw = sch.get_loops(blk)
            sch.reorder(n, oc, oh, ow, ic, kh, kw, ob)
            main_lp = sch.fuse(n, oc, oh, ow)
            bx, ty, tx = sch.split(main_lp, [None, 16, 16])
            sch.bind(tx, "threadIdx.x")
            sch.bind(ty, "threadIdx.y")
            sch.bind(bx, "blockIdx.x")

            ico, icv = sch.split(ic, [None, 4])
            sch.reorder(ico, kh, kw, icv, ob)
            rblk = sch.cache_read(blk, 0, "local")
            sch.compute_at(rblk, kw)
            sch.vectorize(sch.get_loops(rblk)[-1])
            wblk = sch.cache_write(blk, 0, "local")
            sch.reverse_compute_at(wblk, tx)
            sch.vectorize(sch.get_loops(wblk)[-1])
            sch.vectorize(ob)
            init_blk = sch.decompose_reduction(blk, ico)
            sch.vectorize(sch.get_loops(init_blk)[-1])

        def is_data_pad(block: tir.stmt.Block):
            return is_spatial_block(sch, block) and tir.analysis.has_if_then_else(sch.get(block))

        def schedule_conv2d_blocks():

            # Do analysis to find block type
            blocks = sch.get_child_blocks(root_block)
            passed_reduction = False
            for blk in blocks:
                if is_reduction_block(sch, blk):
                    schedule_conv2d(blk)
                    passed_reduction = True
                elif is_data_pad(blk):
                    schedule_data_pad(blk)
                elif is_spatial_block(sch, blk):
                    try:
                        if not passed_reduction:
                            sch.compute_inline(blk)
                        else:
                            sch.reverse_compute_inline(blk)
                    except:  # pylint: disable=W0702
                        pass
                else:
                    raise TypeError("Can't Schedule this Block", sch.get(blk))

        schedule_conv2d_blocks()
        return sch
