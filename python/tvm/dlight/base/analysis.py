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

from tvm import ir, tir
from tvm._ffi import get_global_func
from tvm.target.target import Target
from tvm.tir import Schedule, IterVar, IndexMap
from tvm.tir.schedule import BlockRV
from tvm.tir.function import TensorIntrin
from tvm.tir.schedule.analysis import get_auto_tensorize_mapping_info


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


def normalize_with_tensorcore(
    sch: tir.Schedule, block: BlockRV, intrin_group: Dict[str, str]
) -> tir.Schedule:
    """
    Normalizing the PrimFunc to TensorCore MatMul (for example conv2d with im2col), the python implementation of MetaSchedule::TransformWithTensorIntrin

    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(4, 16, 16, 3, 3, 64):
        with T.block("PadInput_reindex_reindex"):
            v0, v1, v2, v3, v4, v5 = T.axis.remap("SSSSSS", [ax0, ax1, ax2, ax3, ax4, ax5])
            T.reads(PadInput[v0, v1 + v3, v2 + v4, v5])
            T.writes(PadInput_reindex[v0 * 256 + v1 * 16 + v2, v3 * 192 + v4 * 64 + v5])
            PadInput_reindex[v0 * 256 + v1 * 16 + v2, v3 * 192 + v4 * 64 + v5] = PadInput[v0, v1 + v3, v2 + v4, v5]
    for ax0, ax1, ax2, ax3 in T.grid(64, 3, 3, 64):
        with T.block("weight_reindex_reindex"):
            v0, v1, v2, v3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(weight[v1, v2, v3, v0])
            T.writes(weight_reindex[v1 * 192 + v2 * 64 + v3, v0])
            weight_reindex[v1 * 192 + v2 * 64 + v3, v0] = weight[v1, v2, v3, v0]
    """
    desc = TensorIntrin.get(intrin_group["compute"]).desc
    mapping_info = get_auto_tensorize_mapping_info(sch, block, desc)
    assert len(mapping_info.mappings) == 1
    index_map = mapping_info.mappings[0]

    tensor_core_reindex_A = sch.reindex(block, ("read", 0))
    tensor_core_reindex_B = sch.reindex(block, ("read", 1))
    tensor_core_reindex_C = sch.reindex(block, ("write", 0))

    lhs_to_index_map_src = {}
    rhs_to_index_map_tgt = {}
    unmapped_index_map_src = []
    assert len(index_map.initial_indices) == len(mapping_info.lhs_iters)

    for i in range(len(mapping_info.lhs_iters)):
        lhs_to_index_map_src[mapping_info.lhs_iters[i].var] = index_map.initial_indices[i]
    offset = len(index_map.final_indices) - len(mapping_info.rhs_iters)

    for i in range(offset):
        unmapped_index_map_src.append(index_map.final_indices[i])

    for i in range(offset, len(index_map.final_indices)):
        rhs_to_index_map_tgt[mapping_info.rhs_iters[i - offset].var] = index_map.final_indices[i]

    def get_sub_index_map(buffer_region, reindex_block, type: str = "read"):
        # python implementation of
        # src/meta_schedule/schedule_rule/multi_level_tiling_tensor_core.cc:802-822

        sub_index_map_src = []
        sub_index_map_tgt = []

        region = buffer_region.region
        for r_var in region:
            assert r_var.extent == 1
            var_ptr = r_var.min
            assert var_ptr is not None
            lhs_representer = lhs_to_index_map_src[var_ptr]
            sub_index_map_src.append(lhs_representer)
            if lhs_representer in unmapped_index_map_src:
                sub_index_map_tgt.append(lhs_representer)

        original_buffer = sch.get(reindex_block).reads[0].buffer
        if type == "write":
            original_buffer = sch.get(reindex_block).writes[0].buffer
        original_buffer = mapping_info.lhs_buffer_map[original_buffer]
        for i in range(len(mapping_info.rhs_buffer_indices[original_buffer])):
            var = mapping_info.rhs_buffer_indices[original_buffer][i]
            sub_index_map_tgt.append(rhs_to_index_map_tgt[var])
        index_map = IndexMap(sub_index_map_src, sub_index_map_tgt, None)
        print(index_map)
        return index_map

    def transform_layout(
        block: BlockRV, reindex_block: BlockRV, type: Dict[str, int] = ("read", 0)
    ):
        region_str, index = type
        reverse_region_str = {"write": "read", "read": "write"}[region_str]
        if region_str == "read":
            buffer_region = sch.get(block).reads[index]
        else:
            buffer_region = sch.get(block).writes[index]

        reindex_map = get_sub_index_map(buffer_region, reindex_block, region_str)
        sch.transform_layout(reindex_block, (reverse_region_str, 0), reindex_map)

    transform_layout(block, tensor_core_reindex_A, ("read", 0))
    transform_layout(block, tensor_core_reindex_B, ("read", 1))
    transform_layout(block, tensor_core_reindex_C, ("write", 0))

    sch.transform_block_layout(block, index_map)

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

        return tags

    (main_block,) = reduction_blocks
    if _can_be_tensorized(sch, main_block) is None:
        return func, None

    minimal_tensorize_threshold = 64
    block_stmt = sch.get(main_block)
    if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
        apply_tensorization: bool = True
        # the batch dimension is not taken into consideration.
        for item_var in block_stmt.iter_vars[1:]:
            extent = item_var.dom.extent
            if isinstance(extent, tir.expr.IntImm):
                if extent.value <= minimal_tensorize_threshold:
                    apply_tensorization = False
        if apply_tensorization:
            in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
            try:
                intrin_group = get_wmma_intrin_group(
                    load_scope="shared",
                    store_scope="shared",
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    trans_b=True,
                )
            except:
                print("[WARNING] Cannot find the corresponding wmma intrin group")
                return None

            # reindex and transform functions
            sch = normalize_with_tensorcore(sch, main_block, intrin_group)
            tags = analysis_tensorcore_tags(sch, main_block, target)
            return sch.mod["main"], tags

    return func, None
