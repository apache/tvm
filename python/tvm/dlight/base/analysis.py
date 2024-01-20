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
from typing import List, Optional, Set, Union, Tuple, Dict
from typing_extensions import Literal
from dataclasses import dataclass
from enum import Enum

from tvm import ir, tir
from tvm.ir import Range
from tvm.tir.analysis import undefined_vars
from tvm._ffi import get_global_func
from tvm.target.target import Target
from tvm.tir import Schedule, IterVar, Var, PrimExpr
from tvm.tir.schedule import BlockRV


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    simplified : bool
        Whether to use simplified index map (e.g. remove constant axes)

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            if iter_var.iter_type == tir.IterVar.CommReduce:
                # for simplified case (e.g. 1x1 conv kernel)
                kind = IterKind.kIter_K
            else:
                kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    simplified : bool
        Whether to use simplified index map (e.g. remove constant axes)

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    def get_ordered_axes(region: List[Range]) -> Set[Var]:
        axes: List[Var] = []
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes.append(r.min)
        return axes

    def is_common_reduce(var: Var) -> bool:
        for iter_var in block.iter_vars:
            if iter_var.var == var and iter_var.iter_type == IterVar.CommReduce:
                return True
        return False

    def check_last_trait(region: List[Range]):
        axes = get_ordered_axes(region)
        return is_common_reduce(axes[-1])

    A_index_map = make_iter_fusion_index_map(
        A_traits,
        [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
        if check_last_trait(block.reads[0].region)
        else [IterKind.kIter_S, IterKind.kIter_K, IterKind.kIter_I],
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits,
        [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
        if check_last_trait(block.reads[1].region)
        else [IterKind.kIter_S, IterKind.kIter_K, IterKind.kIter_J],
    )

    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].buffer.dtype
    out_dtype = block.writes[0].buffer.dtype
    return (in_dtype, out_dtype)


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
                        dom=iter.dom,
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


def collect_vars_used_in_access_region(region: List[ir.Range]) -> Set[tir.Var]:
    """Collect the variables used in the access region of a buffer region."""
    tir_vars: Set[tir.Var] = set()

    def _collect_tir_var(expr):
        if isinstance(expr, tir.Var):
            tir_vars.add(expr)

    for expr in region:
        assert expr.extent == 1
        tir.stmt_functor.post_order_visit(expr.min, _collect_tir_var)
    return tir_vars


def detect_dominant_read(block: tir.Block) -> tir.PrimExpr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_vars_used_in_access_region(buffer_region.region)
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
        tir_vars = collect_vars_used_in_access_region(buffer_region.region)
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False


def get_reduction_blocks(
    sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]
) -> List[tir.schedule.BlockRV]:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) == 0:
        return None
    return reduction_blocks


def normalize_with_tensorcore(sch: tir.Schedule, main_block: BlockRV) -> Optional[tir.Schedule]:
    block_stmt = sch.get(main_block)

    index_maps = get_index_map(block_stmt)
    if index_maps is None:
        print("[WARNING] Cannot find the appropriate index map for tensorcore")
        return None

    matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

    # `skip_simplify` to  avoid the bug in the 1x1 conv
    block = sch.reindex(main_block, ("read", 0), skip_simplify=True)
    sch.transform_layout(block, ("write", 0), a_index_map)
    block = sch.reindex(main_block, ("read", 1), skip_simplify=True)
    sch.transform_layout(block, ("write", 0), b_index_map)
    block = sch.reindex(main_block, ("write", 0), skip_simplify=True)
    sch.transform_layout(block, ("read", 0), c_index_map)
    sch.transform_block_layout(main_block, matmul_index_map)
    sch.mod["main"] = sch.mod["main"].with_attr("dlight.tensorcore_prenormlized", True)
    return sch


def get_tensorized_func_and_tags(
    func: tir.PrimFunc,
    target: Target,
) -> Tuple[tir.PrimFunc, Dict[str, Union[List[int], int]]]:
    from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
        get_wmma_intrin_group,
    )

    """
        transform function to matmul if necessary (e.g. transform conv2d with im2col)
    """
    # step1. detect whether the function can utilize tensorcore
    sch = tir.Schedule(func)
    root_block = get_root_block(sch)
    blocks = sch.get_child_blocks(root_block)
    reduction_blocks = get_reduction_blocks(sch, blocks)
    if not reduction_blocks or len(reduction_blocks) != 1:
        return func, None

    def _can_be_tensorized(sch: tir.Schedule, block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        conditions = []
        conditions.append(len(block_stmt.reads) == 2)
        conditions.append(len(block_stmt.writes) == 1)
        conditions.append(len(collect_vars_used_in_access_region(block_stmt.writes[0].region)) > 0)
        if not all(conditions):
            return False
        return True

    # step2. transform function to tensorcore matmul (e.g. conv2d with im2col)
    def check_sm_version(arch: str) -> int:
        sm_version = arch.replace("sm_", "")
        return int(sm_version) if sm_version.isdigit() else -1

    def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
        """
        Detect In/Out data types for the given block based on the analysis if read/write buffers.
        """
        assert len(block.reads) > 0 and len(block.writes) > 0
        in_dtype = block.reads[0].buffer.dtype
        out_dtype = block.writes[0].buffer.dtype
        return (in_dtype, out_dtype)

    def analysis_tensorcore_tags(sch: tir.Schedule, block: BlockRV, target: Target) -> bool:
        tags: Dict[str, Union[List[int], int]] = {}
        block_stmt = sch.get(block)

        # analysis tensorcore axis
        # todo(lei): maybe we can remove this in the future
        (write_buffer_region,) = block_stmt.writes
        out_axis = len(write_buffer_region.buffer.shape)
        tags["tensorcore_config"] = [out_axis - 2, out_axis - 1]

        # analysis pipeline stage
        # todo(lei): maybe we can integrate this into policy in the future
        tags["pipeline_stage"] = 1
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 80:
            # enable pipleline stage only for sm_80 devices
            tags["pipeline_stage"] = 2

        # analysis async copy
        # todo(lei): maybe we can integrate this into policy in the future
        tags["use_async_copy"] = 0
        if tags["pipeline_stage"] == 2 and check_sm_version(target.arch) >= 80:
            # async copy only works in software pipeline.
            tags["use_async_copy"] = 1

        # analysis intrin infomation
        def get_ordered_axes(region: List[Range]) -> Set[Var]:
            axes: List[Var] = []
            for r in region:
                if not _is_one(r.extent):
                    raise ValueError("Expect elemwise block access")
                axes.append(r.min)
            return axes

        def is_common_reduce(var: Var) -> bool:
            for iter_var in block_stmt.iter_vars:
                if iter_var.var == var and iter_var.iter_type == IterVar.CommReduce:
                    return True
            return False

        def check_last_trait(region: List[Range]):
            axes = get_ordered_axes(region)
            return is_common_reduce(axes[-1])

        intrin_info: dict = {}
        in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
        intrin_info["in_dtype"] = in_dtype
        intrin_info["out_dtype"] = out_dtype
        # if the last dimension is reduce axis, the B is transposed
        intrin_info["trans_b"] = check_last_trait(block_stmt.reads[1].region)

        tags["intrin_info"] = intrin_info

        return tags

    (main_block,) = reduction_blocks
    if _can_be_tensorized(sch, main_block) is None:
        return func, None

    minimal_tensorize_threshold = 64
    block_stmt = sch.get(main_block)
    if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
        in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
        try:
            _ = get_wmma_intrin_group(
                in_dtype=in_dtype,
                out_dtype=out_dtype,
            )
        except:
            print("[FastDlight][WARNING] Cannot find the corresponding wmma intrin group")
            return func, None

        # reindex and transform functions
        # Normalize tensor functions to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # or C[S, I, J] += A[S, I, K] * B[S, K, J]
        sch = normalize_with_tensorcore(sch, main_block)
        if sch is None:
            return func, None

        block_stmt = sch.get(main_block)
        # the batch dimension is not taken into consideration.
        for item_var in block_stmt.iter_vars[1:]:
            extent = item_var.dom.extent
            if isinstance(extent, tir.expr.IntImm):
                if extent.value < minimal_tensorize_threshold:
                    return func, None
        tags = analysis_tensorcore_tags(sch, main_block, target)
        return sch.mod["main"], tags

    return func, None
