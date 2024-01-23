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
"""A GEMM schedule rule for GPU operators."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Union, Tuple, Dict

from tvm import tir
from tvm.ir import Range
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV
from ..base.analysis import collect_block_iter_vars_used_in_access_region
from tvm.target.target import Target


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


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


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
    skip_blocks: Optional[List[tir.schedule.BlockRV]] = None,
):
    skip_blocks = skip_blocks or []
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            if any(sch.get(producer) == sch.get(skip_block) for skip_block in skip_blocks):
                continue
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)


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


def get_index_map(
    block: tir.Block, layout: List[str] = ["n", "t", "n"]
) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    layout : List[str]
        the target layout index map to be used.
        'n' for [i, k] layout
        't' for [k, j] layout
        'a' for auto inference based on whether the last axis is reduction.

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        print("[WARNING] traits is None, the block is", block)
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

    def infer_layout(layout: str, region: List[Range], kind: str = "A"):
        """
        Infer the layout based on the region and the kind of buffer
        kind: "A", "B", "C"
        """
        primary_iter, secondary_iter, reduction_iter = {
            "A": (IterKind.kIter_I, IterKind.kIter_K, IterKind.kIter_K),
            "B": (IterKind.kIter_K, IterKind.kIter_J, IterKind.kIter_K),
            "C": (IterKind.kIter_I, IterKind.kIter_J, None),
        }[kind]

        spatial_iter = {
            "A": IterKind.kIter_I,
            "B": IterKind.kIter_J,
            "C": None,
        }[kind]

        if layout == "n":
            return [IterKind.kIter_S, primary_iter, secondary_iter]
        elif layout == "t":
            return [IterKind.kIter_S, secondary_iter, primary_iter]
        elif layout == "a":
            # auto inference layout
            # for buffer with reduction axis, we put it as the last axis
            # otherwise, we put it as the first axis
            if kind == "C":
                return [IterKind.kIter_S, primary_iter, secondary_iter]
            else:
                return (
                    [IterKind.kIter_S, spatial_iter, reduction_iter]
                    if check_last_trait(region)
                    else [IterKind.kIter_S, reduction_iter, spatial_iter]
                )
        else:
            raise ValueError(f"Unknown layout {layout}")

    A_index_map = make_iter_fusion_index_map(
        A_traits, infer_layout(layout[0], block.reads[0].region, kind="A")
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, infer_layout(layout[1], block.reads[1].region, kind="B")
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, infer_layout(layout[2], block.writes[0].region, kind="C")
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


def get_reduction_blocks(sch, blocks) -> Optional[List[BlockRV]]:
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


def get_dequantize_block(sch, blocks) -> Optional[BlockRV]:
    # check at least two input and one output
    # at lease one input has uint dtype, and the output dtype is float
    def is_dequantize(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        if len(block_stmt.reads) < 2:
            return False
        has_uint_input = any("uint" in str(region.buffer.dtype) for region in block_stmt.reads)
        if not has_uint_input:
            return False
        if len(block_stmt.writes) != 1 or "float" not in str(block_stmt.writes[0].buffer.dtype):
            return False
        return True

    dequantize_blocks = [block for block in blocks if is_dequantize(block)]
    return dequantize_blocks[0] if len(dequantize_blocks) == 1 else None


def is_identity_or_transpose_block(block_stmt: tir.Block) -> bool:
    iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
    if iter_types != {IterVar.DataPar}:
        return False, False
    if not isinstance(block_stmt.body, tir.BufferStore):
        return False, False
    if not isinstance(block_stmt.body.value, tir.BufferLoad):
        return False, False

    def get_access_vars(region: List[Range]) -> List[Var]:
        axes: List[Var] = []
        for r in region:
            if not _is_one(r.extent):
                return None
            axes.extend(undefined_vars(r.min))
        # remove trivial axis
        trivial_vars = set(
            iter_var.var for iter_var in block_stmt.iter_vars if _is_one(iter_var.dom.extent)
        )
        axes = [axis for axis in axes if axis not in trivial_vars]
        # remove duplicate axis
        axes = [var for i, var in enumerate(axes) if i == 0 or var != axes[i - 1]]
        return axes

    lhs_access_vars = get_access_vars(block_stmt.reads[0].region)[-2:]
    rhs_access_vars = get_access_vars(block_stmt.writes[0].region)[-2:]
    is_identity = list(lhs_access_vars) == list(rhs_access_vars)
    is_transpose = list(lhs_access_vars) != list(rhs_access_vars) and set(lhs_access_vars) == set(
        rhs_access_vars
    )
    return is_identity, is_transpose


def is_identity_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[0]


def is_transpose_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[1]


def inline_transpose_block(sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]):
    result_blocks = []
    for block in blocks:
        if not is_transpose_block(sch.get(block)):
            result_blocks.append(block)
            continue
        try:
            sch.compute_inline(block)
        except:
            try:
                sch.reverse_compute_inline(block)
            except:
                result_blocks.append(block)
    return result_blocks


def normalize_to_matmul(
    sch: tir.Schedule, main_block: BlockRV, layout: List[str] = ["n", "t", "n"]
) -> Optional[tir.Schedule]:
    block_stmt = sch.get(main_block)

    # let layout be 'a' to auto inference the layout
    index_maps = get_index_map(block_stmt, layout=layout)
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
        conditions.append(
            len(
                collect_block_iter_vars_used_in_access_region(
                    block_stmt, block_stmt.writes[0].region
                )
            )
            > 0
        )
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
        sch = normalize_to_matmul(sch, main_block, ["a", "a", "a"])
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
