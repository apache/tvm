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
"""Config definition for schedule"""
from typing import Dict, List, Optional, Tuple

import numpy as np


class TensorCoreExtraConfig:
    """
    This class is used to store extra information for tensorcore
    """
    def __init__(self, AS_shape:Tuple[int], BS_shape:Tuple[int], AF_shape:Tuple[int], BF_shape:Tuple[int], tc_axis:Tuple[int]) -> None:
        self.AS_shape:Tuple[int] = AS_shape
        self.BS_shape:Tuple[int] = BS_shape
        self.AF_shape:Tuple[int] = AF_shape
        self.BF_shape:Tuple[int] = BF_shape
        self.tc_axis:Tuple[int] = tc_axis

class Stride:
    """
    Manages stride information for a given axis of a tensor.
    """
    def __init__(self, stride: int = 1, ax: int = -1) -> None:
        # which axis to put stride on
        self._ax:int = int(ax)
        # the stride size of the axis
        self._stride:int = int(stride)

    @property
    def ax(self) -> int:
        return self._ax

    @property
    def stride(self) -> int:
        return self._stride

    def compute_strides_from_shape(self, shape: List[int]) -> List[int]:
        ndim = len(shape)
        strides = [1 for _ in shape]
        for i in range(ndim - 2, -1, -1):
            if i == self.ax:
                strides[i] = self.stride
            else:
                strides[i] = int(strides[i + 1] * shape[i + 1])
        return strides

    def compute_elements_from_shape(self, shape: List[int]) -> int:
        original_shape = np.prod(shape)
        if not self.is_valid():
            strided_elem = original_shape
        else:
            assert self.ax < len(shape)
            strided_elem = np.prod(shape[0:self.ax+1]) * self.stride
            assert strided_elem >= original_shape
        return int(strided_elem)

    def is_valid(self) -> bool:
        return self.ax >= 0

    def __repr__(self) -> str:
        return f"<Stride, {self._ax}, {self._stride}>"

class TileDict:
    """
    Manages tiling information and configurations for computational tasks.
    """
    def __init__(self, output_tile) -> None:
        self.output_tile = output_tile
        # schedule config
        self.tile_map = {}
        self.rstep_map = {}
        self.cached_tensors_map = {}
        self.output_strides_map = {}
        self.tensor_strides_map = {}
        self.use_cutlass_mma = {}

        # analysis
        self.traffic = -1
        self.smem_cost = -1
        self.block_per_SM = -1
        self.num_wave = -1
        self.grid_size = -1
        self.valid = True

    def get_tile(self, func) -> List[int]:
        return self.tile_map[func]

    def get_rstep(self, func) -> Dict[str, int]:
        return self.rstep_map

    def __hash__(self) -> int:
        return hash(tuple(self.output_tile))


class Config:
    """
    Central configuration class for managing various parameters of computational tasks.
    """
    def __init__(self) -> None:
        self.arch = None
        self.use_tc = None # todo(lei): this should be renamed.
        self.compute_capability = None

        # spacial axes tiling info
        self.block = []
        self.thread = []
        # special axes for tensorCore
        self.warp = []
        self.wmma = []
        self.tc_extra_conf: Optional[TensorCoreExtraConfig] = None
        # reduce axes tiling info
        self.rstep = []
        self.reduce_thread = []
        self.raster_factor = 0
        self.cached_tensors = []
        self.block_order = None
        self.output_strides = {}
        self.schedule_stages = None

        # Experimental
        self._raxis_order = []
        self._step = []
        self.vectorize : Dict[str, int] = {}
        self.pipeline_stage = 1
        self.use_async = False

    def to_dict(self) -> Dict:
        dic = {}
        dic["block"] = self.block
        if self.use_tc:
            dic["warp"] = self.warp
            dic["wmma"] = self.wmma
        else:
            dic["thread"] = self.thread
        dic["rstep"] = self.rstep
        if np.prod(self.reduce_thread) > 1:
            dic["reduce_thread"] = self.reduce_thread
        if self.block_order is not None:
            dic["block_order"] = self.block_order
        if self.use_tc:
            dic["use_tc"] = self.use_tc
        if self.output_strides:
            dic["strides"] = {}
            for k, stride in self.output_strides.items():
                if stride.is_valid():
                    dic["strides"][k] = stride
            if len(dic["strides"]) == 0:
                del dic["strides"]
        if np.prod(self._step) > 1:
            dic["step"] = self._step
        if self._raxis_order != []:
            dic["raxis_order"] = self._raxis_order
        if self.vectorize != {}:
            dic["vectorize"] = self.vectorize
        return dic

    def from_dict(self, dic: Dict) -> "Config":
        self.__init__()
        if "use_tc" in dic:
            self.use_tc = dic["use_tc"]
        self.block = dic["block"]
        if self.use_tc:
            self.warp = dic["warp"]
            self.wmma = dic["wmma"]
        else:
            self.thread = dic["thread"]
        self.rstep = dic["rstep"]
        if "reduce_thread" in dic:
            self.reduce_thread = dic["reduce_thread"]
        else:
            self.reduce_thread = [1 for _ in self.rstep]
        if "block_order" in dic:
            self.block_order = dic["block_order"]
        if "strides" in dic:
            self.output_strides = dic["strides"]
        if "step" in dic:
            self._step = dic["step"]
        if "raxis_order" in dic:
            self._raxis_order = dic["raxis_order"]
        if "vectorize" in dic:
            self.vectorize = dic["vectorize"]
        return self

    @property
    def raxis_order(self) -> List[int]:
        if self._raxis_order != []:
            return self._raxis_order
        return list(range(len(self.rstep)))

    @property
    def step(self) -> List[int]:
        if self._step != []:
            return self._step
        return [1 for _ in self.block]

    def __repr__(self) -> str:
        return str(self.to_dict())

    def complete_config(self, node):
        if not self.use_tc:
            return self
        _, _, wmma_k = self.wmma

        tc_axis = node.infer_tensorcore_axis()

        shapes = node.propogate_reduction_inputs(self.block, {x : self.rstep[0] for x in node.raxis})
        AS_shape, BS_shape = shapes.values()

        shapes = node.propogate_reduction_inputs(self.warp, {x : wmma_k for x in node.raxis})
        AF_shape, BF_shape = shapes.values()

        self.tc_extra_conf = TensorCoreExtraConfig(AS_shape, BS_shape, AF_shape, BF_shape, tc_axis)
        return self
