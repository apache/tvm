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
"""Policy for tensorcore schedule"""
import tvm
from typing import Dict, List, Tuple
import numpy as np

from ..arch import Arch
from ..config import Config, Stride, TileDict, IntrinInfo
from ..node import PrimFuncNode
from .common import coalesced_factor, factorize, get_all_factors
from .default import DefaultPolicy
from ..rasterization import *


class TensorCorePolicy(DefaultPolicy):
    def __init__(self, func: tvm.tir.PrimFunc, arch: Arch, tags: Dict = {}) -> None:
        super().__init__(func, arch, tags)
        # this is the trick for wmma.
        # However, for int8 mma, the wmma_k should be 32.
        self.wmma_k = 16
        self.pipeline_stage: int = 1
        self.use_async_copy: bool = False
        self._legalize_info()

    def _legalize_info(self):
        pipleline_stage = self.prim_func_node.get_tag("pipeline_stage")
        if pipleline_stage:
            self.pipeline_stage = pipleline_stage
        else:
            if self.arch.compute_capability == "sm_80":
                self.pipeline_stage = 2
            else:
                self.pipeline_stage = 1
        use_async_copy = self.prim_func_node.get_tag("use_async_copy")
        if use_async_copy:
            self.use_async_copy = use_async_copy
        else:
            if self.arch.compute_capability == "sm_80":
                self.use_async_copy = 2
            else:
                self.use_async_copy = 1

    def _compute_tc_strides(
        self, node: PrimFuncNode, tile: List[int], rstep: Dict[str, int] = {}
    ) -> Tuple[Stride, Stride, Stride]:
        # strides was used for shared memory padding. which is necessary for avoiding
        # shared memory load bank conflict when we do not applying tensorcore layout.
        shapes = node.propogate_reduction_inputs(tile, rstep)
        AS_shape, BS_shape = shapes.values()
        CS_shape = tile
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()

        # applying strides
        # TODO(leiwang1999): offset should be dynamically set. we can use tag -> enable_offset to control this option..
        offset = 8
        A_high_ax = min(A_ax_m, A_ax_k)
        B_high_ax = min(B_ax_n, B_ax_k)
        C_high_ax = min(C_ax_m, C_ax_n)
        A_stride = Stride(stride=np.prod(AS_shape[A_high_ax + 1 :]) + offset, ax=A_high_ax)
        B_stride = Stride(stride=np.prod(BS_shape[B_high_ax + 1 :]) + offset, ax=B_high_ax)
        C_stride = Stride(stride=np.prod(CS_shape[C_high_ax + 1 :]) + offset, ax=C_high_ax)
        return A_stride, B_stride, C_stride

    def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
        value, cached_tensors = super().infer_node_smem_usage(td, node)
        value *= self.pipeline_stage
        return value, cached_tensors

    def _assign_reduce_step(self, node):
        if not node.get_tag("tensorcore_config"):
            return super()._assign_reduce_step(node)
        result = {}
        for iter_info in node.raxis:
            iter_name = iter_info.var.name
            iter_dom = iter_info.dom.extent
            if iter_dom % 16 > 0:
                result[iter_name] = 16 if iter_dom < 32 else 32  # padding case
            elif iter_dom % 32 == 0:
                result[iter_name] = 32
            else:
                return super()._assign_reduce_step(node)
        return result

    def _expand_reduce_axis(self, td: TileDict):
        # For tensorcore program, if we got a small tilesize, we should consider expand the reduce axis
        # to improve compute efficiency.
        def _check_small_tile(td: TileDict):
            minimal_threadhold = 32
            for node in self.ordered_nodes:
                tile = td.get_tile(node)
                if any([t <= minimal_threadhold for t in tile]):
                    return True
            return False

        if not _check_small_tile(td):
            return None

        smem_limit = min(self.arch.max_smem_usage // td.block_per_SM, self.arch.smem_cap)
        rstep_map = td.rstep_map.copy()

        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)
            # todo(lei): optimzie the all_steps enlarge policy to be a multiple of the original all_steps[k]
            for k in all_steps:
                all_steps[k] = list(filter(lambda x: x % rstep[k] == 0, all_steps[k]))
            if any([v == [] for v in all_steps.values()]):
                return rstep

            def _shared_memory_usage(td: TileDict):
                return node.footprint(td.output_tile, new_rstep_map, td.tensor_strides_map[node])

            def _score(rstep_id):
                rstep = {
                    k.var.name: all_steps[k.var.name][rstep_id[k.var.name]] for k in node.raxis
                }
                score = 0
                shape = node.propogate_inputs(td.get_tile(node), rstep=rstep)
                for i, input_buffer in enumerate(node.input_buffers):
                    score += coalesced_factor(shape[i], input_buffer.shape)
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return None
                return max(candidates, key=lambda x: x[1])[0]

            cur_rstep_id = {
                k.var.name: all_steps[k.var.name].index(rstep[k.var.name]) for k in node.raxis
            }
            new_rstep_map = rstep_map.copy()
            while True:
                new_rstep_id = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map = {
                    k.var.name: all_steps[k.var.name][new_rstep_id[k.var.name]] for k in node.raxis
                }
                old_rstep_map = td.rstep_map
                td.rstep_map = new_rstep_map
                smem_usage, _ = _shared_memory_usage(td)
                td.rstep_map = old_rstep_map
                if smem_usage > smem_limit:
                    break
                else:
                    cur_rstep_id = new_rstep_id
            rstep = {
                k.var.name: all_steps[k.var.name][cur_rstep_id[k.var.name]] for k in node.raxis
            }
            return rstep

        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map)
                rstep_map = rstep
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        return

    def get_node_reduce_step_candidates(self, node):
        if not node.get_tag("tensorcore_config"):
            return super().get_node_reduce_step_candidates(node)
        else:
            # must be a a multiple of wmma_k
            return {
                k.var.name: [
                    x * self.wmma_k for x in get_all_factors(int(k.dom.extent) // self.wmma_k)
                ]
                for k in node.raxis
            }

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if node.get_tag("tensorcore_config"):
                ax_m, ax_n = node.get_tag("tensorcore_config")
                block_m, block_n = td.tile_map[node][ax_m], td.tile_map[node][ax_n]
                # check the tile size is valid
                wmma_invalid = [
                    block_m < wmma_m or block_n < wmma_n
                    for wmma_m, wmma_n in self.arch.get_avaliable_tensorintrin_shapes()
                ]
                if all(wmma_invalid):
                    return False
                if any([y % x for x, y in zip(td.tile_map[node], node.get_space_dim())]):
                    return False
        return super().check_tile_shape_isvalid(td)

    def _can_implement_layout(self, node: PrimFuncNode, td: TileDict):
        # Not implemented yet
        # This function is used to check whether we can implement swizzling
        # layout under this tile config
        return False

    def compute_node_stride_map(self, node: PrimFuncNode, td: TileDict):
        if not node.get_tag("tensorcore_config"):
            return super().compute_node_stride_map(node, td)
        use_layout = self._can_implement_layout(node, td)

        AS_stride, BS_stride, C_stride = self._compute_tc_strides(
            node, td.get_tile(node), td.get_rstep(node)
        )
        A_stride, B_stride, _ = self._compute_tc_strides(node, td.get_tile(node))
        tensor_strides = {}
        output_strides = {
            int(i + len(node.input_buffers)): Stride() for i, _ in enumerate(node.output_buffers)
        }
        tensor_strides = {}
        # when connected to shared input, should use full stride without rstep
        for i, (stride, stride_full) in enumerate(
            zip([AS_stride, BS_stride], [A_stride, B_stride])
        ):
            if use_layout:
                continue
            _ = node.block_analyzer.get_input_buffers(node.reduction_block)[i].name
        # TODO(lei): should dig further for shared memory connection case.

        return output_strides, tensor_strides

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        if not node.get_tag("tensorcore_config"):
            return super()._assign_block_size(node, td, block_size)
        ax_m, ax_n = node.get_tag("tensorcore_config")
        if block_size % self.arch.warp_size != 0:
            return None
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        warps = block_size // self.arch.warp_size
        ndim = len(tile)

        wmma = self.arch.get_avaliable_tensorintrin_shapes()[-1]
        wmma_tile = [1 for _ in range(ndim)]
        wmma_tile[ax_m] = wmma[0]
        wmma_tile[ax_n] = wmma[1]

        space = [tile[i] // wmma_tile[i] for i in range(ndim)]
        if tile[ax_m] < wmma_tile[ax_m] or tile[ax_n] < wmma_tile[ax_n]:
            # allow pad, otherwise, we can not get a valid tile shape
            return None
        if np.prod(space) % warps != 0:
            return None
        factors = factorize(np.prod(space) // warps)

        def _score(node, thread):  # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propogate_inputs(block_tile)
            for i, _ in enumerate(node.input_buffers):
                score += np.prod(shape[i]) / self.arch.bandwidth[1]
            return score

        warp_tile = wmma_tile.copy()
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if tile[i] % (warp_tile[i] * factor) != 0:
                    continue
                warp_tile[i] *= factor
                score_map[i] = (_score(node, warp_tile), i)
                warp_tile[i] //= factor
            if len(score_map) == 0:
                return None
            dim_order = sorted(score_map.keys(), key=lambda x: score_map[x])
            warp_tile[dim_order[0]] *= factor

        codegen_dict = Config()
        codegen_dict.block = tile
        codegen_dict.warp = warp_tile
        codegen_dict.use_tc = True
        codegen_dict.pipeline_stage = self.pipeline_stage
        codegen_dict.use_async = self.use_async_copy
        codegen_dict.rstep = [int(rsteps[ax.var.name]) for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]
        codegen_dict.rasterization_plan = self.plan_rasterization(td)
        codegen_dict.wmma = wmma + [self.wmma_k]

        intrin_info = node.get_tag("intrin_info")
        if intrin_info:
            codegen_dict.intrin_info = IntrinInfo(**intrin_info)

        codegen_dict.complete_config(node)
        codegen_dict.vectorize = self._plan_vectorize(self.prim_func_node, td, block_size)
        return codegen_dict

    def plan_rasterization(self, td: TileDict):
        conditions = []
        # only support single node for now
        conditions.append(len(self.ordered_nodes) > 1)
        # small op don't need this
        conditions.append(td.num_wave < 4)
        # only on Ampere+ arch
        conditions.append(self.arch.compute_capability < "80")

        def _check_memory_size():
            overall_gmem_size_in_bytes: int = 0
            for node in self.ordered_nodes:
                for arg in node.args:
                    overall_gmem_size_in_bytes += (
                        int(np.prod(arg.shape)) * tvm.DataType(arg.dtype).bits // 8
                    )
            return overall_gmem_size_in_bytes < (self.arch.l2_cache_size_bytes * 4)

        conditions.append(_check_memory_size())
        if any(conditions):
            return NoRasterization()
        # otherwise, simply provide a block rasterization factor
        raster_factor = int(self.arch.compute_max_core**0.5)

        return Rasterization2DColumn(raster_factor)
