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
from typing import List, Optional, Union

from typing_extensions import Literal

from tvm import tir
from tvm._ffi import get_global_func
from tvm.target.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import BlockRV


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: tir.schedule.LoopRV

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tir.Var,
        dom: tir.PrimExpr,
        loop_rv: tir.schedule.LoopRV,
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv

    @property
    def dom(self) -> Union[int, tir.PrimExpr]:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, tir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom})'

    def __repr__(self) -> str:
        return str(self)


class BlockInfo:
    """Information about a TIR block."""

    name: str
    iters: List[IterInfo]
    block_rv: tir.schedule.BlockRV
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        iters: List[IterInfo],
        block_rv: tir.schedule.BlockRV,
        reduction_block: bool = False,
    ):
        """Construct a BlockInfo object."""
        self.name = name
        self.block_rv = block_rv
        self.iters = iters
        self._reduction_block = reduction_block

    def dom(self) -> List[Union[int, tir.PrimExpr]]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the block is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_reduction(self) -> bool:
        """Whether the block is a reduction workload."""
        # TODO(@junrushao): distinguish GEMV and reduction
        return self._reduction_block

    def is_gemv(self) -> bool:
        """Whether the block is a GEMV workload."""
        raise NotImplementedError

    def is_gemm(self) -> bool:
        """Whether the block is a GEMM workload."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f'BlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


_normalize_prim_func = get_global_func("tir.schedule.NormalizePrimFunc")


def normalize_prim_func(sch: tir.Schedule) -> Optional[List[BlockInfo]]:
    """Normalize the primfunc to normal form"""
    try:
        result = _normalize_prim_func(sch)
        if result is None:
            return None
    except Exception:  # pylint: disable=broad-except
        return None

    def _iter_kind(i: tir.IterVar) -> str:
        return {
            tir.IterVar.DataPar: "S",
            tir.IterVar.CommReduce: "R",
        }.get(i.iter_type, "O")

    blocks: List[BlockInfo] = []
    for block, loops, iters, is_reduction in zip(*result):
        blocks.append(
            BlockInfo(
                name=sch.get(block).name_hint,
                iters=[
                    IterInfo(
                        kind=_iter_kind(iter),  # type: ignore
                        var=iter.var,
                        dom=iter.dom.extent,
                        loop_rv=loop,
                    )
                    for loop, iter in zip(loops, iters)
                ],
                block_rv=block,
                reduction_block=is_reduction,
            )
        )
    return blocks


def _assert_gpu_target(target: Target):
    if "gpu" not in target.keys:
        raise ValueError(f"Expect a GPU target, but got {target}")


def get_max_threads_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


def get_max_shared_memory_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_shared_memory_per_block = target.attrs.get("max_shared_memory_per_block", None)
    if max_shared_memory_per_block is None:
        raise ValueError(
            f"Cannot find `max_shared_memory_per_block` in {target}, please specify it manually"
        )
    return int(max_shared_memory_per_block)


def get_root_block(sch: Schedule, func_name: str = "main") -> BlockRV:
    try:
        block = sch.mod[func_name].body.block
    except:
        raise ValueError(
            f"The function body is expected to be the root block, but got:\n"
            f"{sch.mod[func_name].body}"
        )
    return sch.get_block(block.name_hint)
