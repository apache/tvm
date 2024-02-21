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
"""Common schedule strategies for TIR."""
from typing import Callable, List

from tvm import tir

from .analysis import BlockInfo

def get_block(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
    name: str,
):
    """Get the target block from a schedule.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to get target block.
    name : str
        The name of the target block.

    Returns
    -------
    target_block : BlockRV
        The target block.
    """

    target_block : tir.BlockRV = None
    for block_info in blocks:
        block = block_info.block_rv
        if sch.get(block).name_hint == name:
            target_block = block
    return target_block

def get_output_blocks(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
):
    """Get the output blocks of a schedule.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to get output blocks.
    blocks : List[BlockInfo]
        The blocks to be analyzed.

    Returns
    -------
    output_blocks : List[BlockInfo]
        The output blocks.
    """

    # collect arguments buffer
    func = sch.mod["main"]
    args = list(func.buffer_map.values())

    output_blocks = []
    for block_info in blocks:
        block = block_info.block_rv
        for write in sch.get(block).writes:
            if write.buffer in args:
                output_blocks.append(block)
    
    return output_blocks


def try_inline(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
) -> List[BlockInfo]:
    """Try to inline as many blocks as possible, and return the remaining blocks.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to inline blocks.
    blocks : List[BlockInfo]
        The blocks to be inlined.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    def _trial(func: Callable):
        for i, block in enumerate(blocks):
            try:
                func(block.block_rv)
            except:  # pylint: disable=bare-except
                continue
            return i
        return None

    while True:
        i = _trial(sch.compute_inline)
        if i is None:
            i = _trial(sch.reverse_compute_inline)
        if i is None:
            break
        blocks.pop(i)
    return blocks


def try_inline_contiguous_spatial(
    sch: tir.Schedule,
    block_infos: List[BlockInfo],
) -> List[BlockInfo]:
    """Try to inline contiguous spatial blocks in a schedule

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to inline blocks.
    block_infos : List[BlockInfo]
        The blocks to be try.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    if block_infos is None:
        return None
    results = []
    spatial_blocks = []
    block: BlockInfo
    for block in block_infos:
        if block.is_injective():
            spatial_blocks.append(block)
        elif spatial_blocks:
            results.extend(try_inline(sch, spatial_blocks))
            results.append(block)
            spatial_blocks = []
        else:
            results.append(block)
    if spatial_blocks:
        results.extend(try_inline(sch, spatial_blocks))
    return results
