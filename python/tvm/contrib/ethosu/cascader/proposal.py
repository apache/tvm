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
"""Proposal class to hold graph scheduling information."""
from typing import Dict, FrozenSet, List
import tvm._ffi
from tvm.contrib.ethosu.cascader.plan import Plan

from tvm.runtime import Object

from . import _ffi_api
from .graph import Tensor, Part
from .tensor_config import TensorConfig, MemoryRegion


@tvm._ffi.register_object("contrib.ethosu.cascader.Proposal")
class Proposal(Object):
    """Proposal class"""

    def __init__(
        self,
        part_group: FrozenSet[Part],
        plans: List[Plan],
        input_tensor_configs: Dict[Tensor, TensorConfig],
        memory_usage: Dict[MemoryRegion, int],
        cycles: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Proposal,
            list(part_group),
            plans,
            input_tensor_configs,
            memory_usage,
            cycles,
        )

    @property
    def graph(self):
        return self._graph

    @property
    def part_group(self):
        return frozenset(self._part_group)

    @property
    def plans(self):
        return list(self._plans)

    @property
    def input_tensor_configs(self):
        return dict(self._input_tensor_configs)

    @property
    def memory_usage(self):
        return int(self._memory_usage)

    @property
    def cycles(self):
        return int(self._cycles)
