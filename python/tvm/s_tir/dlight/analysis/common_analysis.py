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
# ruff: noqa: E722

# pylint: disable=missing-function-docstring, missing-class-docstring
# pylint: disable=unused-argument, unused-variable
"""Analysis on TIR blocks, loops and functions."""

from collections import namedtuple
from typing import List, Optional, Set, Tuple, Union

from tvm_ffi import get_global_func
from typing_extensions import Literal

from tvm import ir, s_tir, tir
from tvm.runtime import DataType
from tvm.s_tir import Schedule
from tvm.s_tir.schedule import SBlockRV
from tvm.target.target import Target


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: s_tir.schedule.LoopRV

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tir.Var,
        dom: tir.PrimExpr,
        loop_rv: s_tir.schedule.LoopRV,
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


get_sblockrealize = get_global_func("s_tir.schedule.GetSBlockRealize")
# BufferIndex Types
Index = namedtuple("Index", ["sub"])  # c
RemIndex = namedtuple("RemIndex", ["sub", "div"])  # c%len
DivIndex = namedtuple("DivIndex", ["sub", "div"])  # c//len
MergeIndex = namedtuple("MulIndex", ["dom", "mul", "sub"])  # co*len + cb
BufIndex = List[Union[Index, RemIndex, DivIndex, MergeIndex, None]]


class BufferInfo:
    "Information about Buffer. Provides useful analysis"

    buf_region: tir.BufferRegion
    shape: Tuple[int]
    assoc_lps: List[Union[s_tir.schedule.LoopRV, None]]
    assoc_lps_info: List[Union[tir.For, None]]

    def __init__(
        self,
        sch: s_tir.Schedule,
        block_rv: s_tir.schedule.SBlockRV,
        buf_region: tir.BufferRegion,
        lps: Union[List[s_tir.schedule.LoopRV], None],
    ):
        block = sch.get(block_rv)
        if lps is None:
            lps = sch.get_loops(block_rv)
        loops = [sch.get(lp) for lp in lps]
        iter_vars = [Var.var for Var in block.iter_vars]
        iter_values = get_sblockrealize(sch, block_rv).iter_values
        lpvar_lp = dict([loop.loop_var, lp] for loop, lp in zip(loops, lps))
        var_lp = dict(zip(iter_vars, [lpvar_lp.get(val, None) for val in iter_values]))

        def extract_index_types(buf: tir.BufferRegion) -> BufIndex:
            buf_index = []
            for expr in buf.region:
                expr = expr.min
                dim = None
                if isinstance(expr, tir.expr.Add) and isinstance(expr.b, tir.expr.Var):
                    var_add = expr.b
                    if (
                        isinstance(expr, tir.expr.Mul)
                        and isinstance(expr.a, tir.expr.Var)
                        and isinstance(expr.b, tir.expr.IntImm)
                    ):
                        mul = expr.b
                        var_mul = expr.a
                        dim = MergeIndex(var_mul, mul, var_add)
                elif (
                    isinstance(expr, tir.expr.FloorMod)
                    and isinstance(expr.a, tir.expr.Var)
                    and isinstance(expr.b, tir.expr.IntImm)
                ):
                    dim = RemIndex(expr.a, expr.b)
                elif (
                    isinstance(expr, tir.expr.FloorDiv)
                    and isinstance(expr.a, tir.expr.Var)
                    and isinstance(expr.b, tir.expr.IntImm)
                ):
                    dim = DivIndex(expr.a, expr.b)
                elif isinstance(expr, tir.expr.Var):
                    dim = Index(expr)
                buf_index.append(dim)
            return buf_index

        indexes = extract_index_types(buf_region)
        assoc_lps = [
            (
                var_lp.get(getattr(idx, "sub"), None)
                if not isinstance(idx, DivIndex) and idx is not None
                else None
            )
            for idx in indexes
        ]

        self.buf_region = buf_region
        self.assoc_lps = assoc_lps
        self.assoc_lps_info = [(sch.get(lp) if lp is not None else None) for lp in assoc_lps]
        self.shape = buf_region.buffer.shape

    def get_scope(self) -> str:
        return self.buf_region.buffer.scope()

    def get_vecsize(self, buf_index: int = 0, vbits: int = 128):
        if self.assoc_lps_info[-1] is None:
            return None

        vlp_extent = int(self.assoc_lps_info[-1].extent) & ~(
            int(self.assoc_lps_info[-1].extent) - 1
        )
        vbuf_extent = int(self.shape[-1]) & ~(int(self.shape[-1]) - 1)

        return min(vlp_extent, vbuf_extent, vbits // DataType(self.buf_region.buffer.dtype).bits)

    def __str__(self) -> str:
        return f"BufferInfo({self.buf_region})"

    def __repr__(self) -> str:
        return str(self)


class SBlockInfo:
    """Information about a TIR block."""

    name: str
    iters: List[IterInfo]
    block_rv: s_tir.schedule.SBlockRV
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        iters: List[IterInfo],
        block_rv: s_tir.schedule.SBlockRV,
        reduction_block: bool = False,
    ):
        """Construct a SBlockInfo object."""
        self.name = name
        self.block_rv = block_rv
        self.iters = iters
        self._reduction_block = reduction_block

    def dom(self) -> List[Union[int, tir.PrimExpr]]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def read_bufs(self, sch: s_tir.Schedule) -> List[BufferInfo]:
        block_stmt = sch.get(self.block_rv)
        lps = sch.get_loops(self.block_rv)
        return [BufferInfo(sch, self.block_rv, buf, lps) for buf in block_stmt.reads]

    def write_bufs(self, sch: s_tir.Schedule) -> List[BufferInfo]:
        block_stmt = sch.get(self.block_rv)
        lps = sch.get_loops(self.block_rv)
        return [BufferInfo(sch, self.block_rv, buf, lps) for buf in block_stmt.writes]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the SBlock is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock is elementwise, i.e. trivial mapping between read/write region"""

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
            if not _check_unit_var_range(r_dom, var) or not _check_unit_var_range(w_dom, var):
                return False
        return True

    def get_loops(self) -> List[s_tir.schedule.LoopRV]:
        return [iter_info.loop_rv for iter_info in self.iters]

    def is_reduction(self) -> bool:
        """Whether the SBlock is a reduction workload."""
        # TODO(@junrushao): distinguish GEMV and reduction
        return self._reduction_block

    def is_layout_transform(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock can be considered having a Layout Transform Pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_bufs(sch)) == 1
            and len(self.read_bufs(sch)) == 1
            and not self.is_elementwise(sch)
            and not get_global_func("s_tir.schedule.HasIfThenElse")(sch.get(self.block_rv))
        )

    def is_data_pad(self, sch: s_tir.Schedule) -> bool:
        """Whether the SBlock can be considered having a data pad pattern"""
        return (
            all(k == "S" for k in self.dom_kind())
            and len(self.write_bufs(sch)) == 1
            and len(self.read_bufs(sch)) == 1
            and not self.is_elementwise(sch)
            and len(self.write_bufs(sch)[0].buf_region.region)
            == len(self.read_bufs(sch)[0].buf_region.region)
            and get_global_func("s_tir.schedule.HasIfThenElse")(sch.get(self.block_rv))
        )

    def is_convolution(self) -> bool:
        """Whether a SBlock can be considered having Convolution Pattern"""
        raise NotImplementedError

    def is_pool(self) -> bool:
        """Whether a SBlock can be considered having Pooling Pattern"""
        raise NotImplementedError

    def is_gemv(self) -> bool:
        """Whether the SBlock is a GEMV workload."""
        raise NotImplementedError

    def is_gemm(self) -> bool:
        """Whether the SBlock is a GEMM workload."""
        raise NotImplementedError

    def __str__(self) -> str:
        return f'SBlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


_normalize_prim_func = get_global_func("s_tir.schedule.NormalizePrimFunc")


def normalize_prim_func(sch: s_tir.Schedule) -> Optional[List[SBlockInfo]]:
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

    blocks: List[SBlockInfo] = []
    for block, loops, iters, is_reduction in zip(*result):
        blocks.append(
            SBlockInfo(
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


def get_sblock_info(sch: s_tir.Schedule, block: s_tir.schedule.SBlockRV) -> SBlockInfo:
    def _iter_kind(loop: tir.IterVar) -> str:
        return {tir.IterVar.DataPar: "S", tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

    def _is_reduction_block(block: s_tir.schedule.SBlockRV):
        for iter_var in sch.get(block).iter_vars:
            if _iter_kind(iter_var) == "R":
                return True
        return False

    return SBlockInfo(
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


def get_root_block(sch: Schedule, func_name: str = "main") -> SBlockRV:
    try:
        block = sch.mod[func_name].body.block
    except Exception:
        raise ValueError(
            f"The function body is expected to be the root block, but got:\n"
            f"{sch.mod[func_name].body}"
        )
    return sch.get_sblock(block.name_hint)


def collect_block_iter_vars_used_in_access_region(
    block: tir.SBlock, region: List[ir.Range]
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


def detect_dominant_read(block: tir.SBlock) -> tir.PrimExpr:
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
    sch: s_tir.Schedule,
    block: s_tir.schedule.SBlockRV,
    epilogue: s_tir.schedule.SBlockRV,
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
