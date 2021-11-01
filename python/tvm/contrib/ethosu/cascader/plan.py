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
"""Plan class to hold subgraph scheduling information."""
from typing import Dict, FrozenSet
import tvm._ffi

from tvm.runtime import Object

from . import _ffi_api
from .graph import Tensor, Part
from .tensor_config import TensorConfig, MemoryRegion


@tvm._ffi.register_object("contrib.ethosu.cascader.Plan")
class Plan(Object):
    """Plan class"""

    def __init__(
        self,
        tensor_configs: Dict[Tensor, TensorConfig],
        open_configs: FrozenSet[TensorConfig],
        output_config: TensorConfig,
        part_group: FrozenSet[Part],
        interior_region: MemoryRegion,
        memory_usage: int,
        cycles: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Plan,
            list(tensor_configs.values()),
            list(open_configs),
            output_config,
            list(part_group),
            interior_region,
            memory_usage,
            cycles,
        )

    def merge(self, other):
        return _ffi_api.PlanMerge(self, other)

    def benchmark_merge(self, other, repeats):
        return _ffi_api.PlanMergeBenchmark(self, other, repeats)

    @property
    def tensor_configs(self):
        tensor_configs = {}
        for config in self._tensor_configs:
            tensor_configs[config.tensor] = config
        return tensor_configs

    @property
    def open_configs(self):
        return frozenset(self._open_configs)

    @property
    def output_config(self):
        return self._output_config

    @property
    def part_group(self):
        return frozenset(self._part_group)

    @property
    def interior_region(self):
        return self._interior_region

    @property
    def memory_usage(self):
        return self._memory_usage

    @property
    def cycles(self):
        return self._cycles

    def __repr__(self):
        return (
            f"Plan(tensor_configs={self.tensor_configs}, "
            f"open_configs={self.open_configs}, "
            f"output_config={self.output_config}, "
            f"part_group={self.part_group}, "
            f"interior_region={self.interior_region.name}, "
            f"memory_usage={self.memory_usage}, "
            f"cycles={self.cycles}, "
        )
