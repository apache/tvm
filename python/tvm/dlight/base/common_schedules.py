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
                func(block.block)
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
