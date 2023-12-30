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
import tvm
from tvm import tir
from tvm.tir import IterVar, Var, PrimFunc
from typing import Any, Iterable, Dict, List, Tuple
import functools
import numpy as np
from tvm.tir.schedule.schedule import BlockRV
from ..analysis import BlockInfo
from .. import analysis
from .. import normalize_prim_func
from .shape_inference import get_analyzer_by_tir


def pre_order_traverse(block_analyzer, blocks, func):
    visited = set()

    def _traverse(block):
        if block in visited:
            return
        visited.add(block)
        for input_blocks in block_analyzer.get_producer_blocks(block):
            _traverse(input_blocks)
        func(block)

    for block in blocks:
        _traverse(block)


class BlockAnalyzer(object):
    def __init__(self, sch) -> None:
        self.sch: tir.Schedule = sch
        self.block_infos: List[BlockInfo] = normalize_prim_func(self.sch)

    def get_reduction_blocks(self, sch, blocks) -> bool:
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

    def get_block_name(self, block: BlockRV) -> str:
        return self.sch.get_sref(block).stmt.name_hint

    def get_block_info(self, block: BlockRV) -> BlockInfo:
        for block_info in self.block_infos:
            if self.get_block_name(block) == block_info.name:
                return block_info
        return None

    def get_spatial_axis(self, block: BlockRV) -> List[IterVar]:
        block_info = self.get_block_info(block)
        axis = []
        for iter in block_info.iters:
            if iter.kind == "S":
                axis.append(iter)
        return axis

    def get_reduce_axis(self, block: BlockRV) -> List[IterVar]:
        block_info = self.get_block_info(block)
        raxis = []
        for iter in block_info.iters:
            if iter.kind == "R":
                raxis.append(iter)
        return raxis

    def get_input_buffers(self, block: BlockRV) -> List[tir.Buffer]:
        buffers = []
        for read in self.sch.get_sref(block).stmt.reads:
            buffers.append(read.buffer)
        return buffers

    def get_output_buffers(self, block: BlockRV) -> List[tir.Buffer]:
        buffers = []
        for write in self.sch.get_sref(block).stmt.writes:
            buffers.append(write.buffer)
        return buffers

    def get_buffers(self, block: BlockRV) -> List[tir.Buffer]:
        return self.get_input_buffers(block) + self.get_output_buffers(block)

    def get_producer_blocks(self, block: BlockRV) -> List[BlockRV]:
        return self.sch.get_producers(block)


class Node(object):
    def __init__(self) -> None:
        self._dtypes = []


class PrimFuncNode(Node):
    def __init__(self, prim_func: PrimFunc, tags: Dict = {}) -> None:
        super().__init__()
        self.prim_func = prim_func
        self.sch: tir.Schedule = tir.Schedule(self.prim_func)
        self.block_analyzer: BlockAnalyzer = BlockAnalyzer(self.sch)
        self.schedule_block = None
        self.blocks: List[BlockRV] = []
        self.output_blocks: List[BlockRV] = None
        self.reduction_block: BlockRV = None
        self.raxis = []
        self.input_buffers = []
        self.output_buffers = []
        self.buffers = []
        self.args = []
        self._tag: Dict = {}
        for tag in tags:
            self.add_tag(tag, tags[tag])
        self._analysis_funcinfo()
        self.ana = get_analyzer_by_tir(self.block_analyzer, self.blocks)

    def _analysis_funcinfo(self):
        root_block = analysis.get_root_block(self.sch)
        blocks = self.sch.get_child_blocks(root_block)
        self.blocks = blocks

        self.output_blocks = self.sch.get_output_blocks(root_block)
        reduction_blocks = self.block_analyzer.get_reduction_blocks(self.sch, blocks)
        if reduction_blocks is None:
            self.reduction_block = None
            # analysis base on the first output block
            self.schedule_block = self.output_blocks[0]
        else:
            # analysis on the last reduction block
            self.reduction_block = reduction_blocks[-1]
            # set raxis
            reduce_block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in reduce_block_info.iters:
                if iter.kind == "R":
                    self.raxis.append(iter)
            self.schedule_block = self.reduction_block

        # collect output buffers
        for output_block in self.output_blocks:
            for write in self.sch.get_sref(output_block).stmt.writes:
                if write not in self.output_buffers:
                    self.output_buffers.append(write.buffer)

        for buffer in self.prim_func.buffer_map.values():
            if buffer not in self.output_buffers:
                self.input_buffers.append(buffer)

        self.args = self.input_buffers + self.output_buffers
        self.buffers = [buffer for buffer in self.prim_func.buffer_map.values()]

        # set dtype
        self.set_dtype(tvm.DataType(self.output_buffers[0].dtype))

    @functools.lru_cache()
    def get_space_dim(self) -> List[int]:
        dim_size = []
        if self.reduction_block:
            block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in block_info.iters:
                if iter.kind == "S":
                    dim_size.append(int(iter.dom))
        else:
            loops = self.sch.get_loops(self.schedule_block)
            for loop in loops:
                sref = self.sch.get_sref(loop)
                dim_size.append(int(sref.stmt.extent))
        return [int(x) for x in dim_size]

    def add_tag(self, k: str, v: Any = True) -> None:
        self._tag[k] = v

    def get_tag(self, k: str) -> Any:
        if k not in self._tag:
            return None
        return self._tag[k]

    def set_dtype(self, dtype: tvm.DataType, id=0) -> None:
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if dtype == tvm.DataType("bool"):
            dtype = tvm.DataType("int8")
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def get_dtype(self, id=0) -> tvm.DataType:
        return self._dtypes[id]

    def get_buffer_dtype(self, buffer: tir.Buffer) -> tvm.DataType:
        return tvm.DataType(buffer.dtype)

    def propogate(self, tile, rstep={}, targets=None):
        shape = {
            buffer.name: [tvm.arith.ConstIntBound(0, val - 1) for val in tile]
            for buffer in self.output_buffers
        }
        return self.ana.infer(shape, rstep, targets)

    def propogate_inputs(self, tile, rstep={}) -> List[List[int]]:
        read_idx_offset = len(self.input_buffers)
        targets = [t.name for t in self.args[:read_idx_offset]]
        shapes, intermediate_bind = self.propogate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[:read_idx_offset]):
            if arg.name in intermediate_bind:
                results.append(shapes[arg.name])
                continue
            # should not exceed original shape
            trimmed_shape = list(map(min, zip(shapes[arg.name], self.input_buffers[i].shape)))
            results.append(trimmed_shape)
        return results

    def propogate_reduction_inputs(self, shape, rstep={}) -> Dict[str, List[int]]:
        if self.reduction_block is None:
            return {}
        targets = [b.name for b in self.block_analyzer.get_input_buffers(self.reduction_block)]
        results, _ = self.propogate(shape, rstep, targets)
        return results

    def get_reduce_inputs_dtype(self):
        if self.reduction_block is None:
            return {}
        return {
            b.name: tvm.DataType(b.dtype)
            for b in self.block_analyzer.get_input_buffers(self.reduction_block)
        }

    @functools.lru_cache()
    def infer_tensorcore_axis(self) -> Tuple[int]:
        # axis is fixed for one expression, so only inference and cached
        assert self.get_tag("tensorcore_config")
        C_ax_m, C_ax_n = self.get_tag("tensorcore_config")
        wmma_m, wmma_n, wmma_k = [16, 16, 16]  # just for testing, any number is ok
        CL_shape = [1] * len(self.get_space_dim())
        CL_shape[C_ax_m] = wmma_m
        CL_shape[C_ax_n] = wmma_n
        shapes = self.propogate_reduction_inputs(CL_shape, {x.var.name: 1 for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_m = A_deps.index(wmma_m)
        B_ax_n = B_deps.index(wmma_n)

        CL_shape = [1] * len(self.get_space_dim())
        shapes = self.propogate_reduction_inputs(CL_shape, {x.var.name: wmma_k for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_k = len(A_deps) - 1 - A_deps[::-1].index(wmma_k)
        B_ax_k = len(B_deps) - 1 - B_deps[::-1].index(wmma_k)
        tc_axis = (A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n)
        return tc_axis

    def footprint(self, shape, rstep, stride_map={}) -> int:
        result = 0
        shapes, _ = self.propogate(shape, rstep)

        def is_broadcast_pattern(buffer, output_buffer):
            return len(shapes[output_buffer.name]) > len(shapes[buffer.name]) and np.prod(
                shapes[output_buffer.name]
            ) > np.prod(shapes[buffer.name])

        def is_after_reduce_stage(block):
            if not self.reduction_block:
                return False
            reduce_dependent_blocks = getattr(self, "reduce_dependent_blocks", None)
            if reduce_dependent_blocks is None:
                reduce_dependent_blocks = set()
                pre_order_traverse(
                    self.block_analyzer,
                    [*self.output_blocks],
                    lambda block: reduce_dependent_blocks.add(block),
                )
                self.reduce_dependent_blocks = reduce_dependent_blocks
            return block not in reduce_dependent_blocks

        # compute cached stages
        cached_tensor = []
        for block in self.output_blocks:
            output_buffer = self.block_analyzer.get_output_buffers(block)[0]
            for buffer in self.block_analyzer.get_input_buffers(block):
                cache = buffer.name not in cached_tensor and (
                    is_broadcast_pattern(buffer, output_buffer)
                    or self.block_analyzer.get_block_info(block).is_reduction
                )
                if not cache:
                    continue
                cached_tensor.append(buffer.name)
                if is_after_reduce_stage(block):
                    continue  # cache after reduce op can often reuse buffer in reduce stage

                if buffer.name in stride_map:
                    num_elem = stride_map[buffer.name].compute_elements_from_shape(
                        shapes[buffer.name]
                    )
                else:
                    num_elem = np.prod(shapes[buffer.name])
                buffer_len = num_elem * int((tvm.DataType(buffer.dtype).bits + 7) // 8)
                buffer_len = (buffer_len + 31) // 32 * 32
                result += buffer_len
        return result, cached_tensor
