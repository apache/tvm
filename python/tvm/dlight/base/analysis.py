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
"""Analysis on TIR blocks, loops and functions."""
from typing import List, Union

from tvm import tir


class BlockInfo:
    """Information about a TIR block."""

    block: tir.schedule.BlockRV
    """The TIR block the current schedule refers to"""
    name: str
    """The name of the block"""
    iters: List[tir.IterVar]
    """The iteration domains of the current block"""

    def __init__(
        self,
        sch: tir.Schedule,
        block: tir.schedule.BlockRV,
    ):
        """Construct a BlockInfo object via TIR schedule."""
        tir_block = sch.get(block)
        self.block = block
        self.name = tir_block.name_hint
        self.iters = list(tir_block.iter_vars)

    def dom(self) -> List[Union[int, tir.PrimExpr]]:
        """The iteration domain of the block."""

        def _iter_dom(i: tir.IterVar) -> Union[int, tir.PrimExpr]:
            result = i.dom.extent
            if isinstance(result, tir.IntImm):
                result = int(result)
            return result

        result = [_iter_dom(i) for i in self.iters]
        return result

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""

        def _iter_kind(i: tir.IterVar) -> str:
            return {
                tir.IterVar.DataPar: "S",
                tir.IterVar.CommReduce: "R",
            }.get(i.iter_type, "O")

        return "".join(_iter_kind(i) for i in self.iters)

    def is_spatial(self) -> bool:
        """Whether the block is spatial, i.e. all its iteration domains are spatial."""
        return all(k == "S" for k in self.dom_kind())

    def __str__(self) -> str:
        return f'BlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)
