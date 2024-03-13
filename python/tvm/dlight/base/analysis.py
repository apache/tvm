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
from typing import List, Optional, Set, Union

from typing_extensions import Literal

from tvm import ir, tir
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

    def is_elementwise(self, sch: tir.Schedule) -> bool:
        """Whether the block is elementwise, i.e. trivial mapping between read/write region"""

        def _check_unit_var_range(dom: ir.Range, var: tir.Var) -> bool:
            return dom.min.same_as(var) and dom.extent == 1

        if not self.is_injective():
            return False
        block = sch.get(self.block_rv)
        if len(block.reads) != 1 or len(block.writes) != 1:
            return False
        r_region = block.reads[0].region
        w_region = block.writes[0].region
        if len(r_region) != len(w_region):
            return False
        for var, r_dom, w_dom in zip(block.iter_vars, r_region, w_region):
            if not _check_unit_var_range(var, r_dom) or not _check_unit_var_range(var, w_dom):
                return False
        return True

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


def collect_block_iter_vars_used_in_access_region(
    block: tir.Block, region: List[ir.Range]
) -> Set[tir.Var]:
    """Collect the block iter variables used in the access region of a buffer region."""
    tir_vars = set()
    for expr in region:
        assert expr.extent == 1
        tir_vars |= collect_vars_used_in_prim_expr(expr.min)
    tir_vars &= set(iter_var.var for iter_var in block.iter_vars)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: tir.PrimExpr) -> Set[tir.Var]:
    """Collect the variables used in the PrimExpr."""
    tir_vars = set()

    def _collect_tir_var(expr):
        if isinstance(expr, tir.Var):
            tir_vars.add(expr)

    tir.stmt_functor.post_order_visit(expr, _collect_tir_var)
    return tir_vars


def detect_dominant_read(block: tir.Block) -> tir.PrimExpr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(block, buffer_region.region)
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    (result,) = dominant_read.buffer.offset_of([e.min for e in dominant_read.region])
    return result


def is_broadcast_epilogue(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
    epilogue: tir.schedule.BlockRV,
) -> bool:
    """Check if the epilogue block is a broadcast pattern"""
    write_buffers = {r.buffer for r in sch.get(block).writes}
    epilogue_iters = {i.var: i for i in sch.get(epilogue).iter_vars if i.dom != 1}
    for buffer_region in sch.get(epilogue).reads:
        if buffer_region.buffer not in write_buffers:
            continue
        tir_vars = collect_block_iter_vars_used_in_access_region(
            sch.get(epilogue), buffer_region.region
        )
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False
