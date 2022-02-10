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
from .graph import Tensor, Part, CascaderGraph
from .tensor_config import TensorConfig, MemoryRegion


@tvm._ffi.register_object("contrib.ethosu.cascader.Proposal")
class Proposal(Object):
    """A class which describes how to schedule a CascaderGraph as a series of disjoint Plans.

    Attributes
    ----------
    graph : CascaderGraph
        The CascaderGraph to which the Proposal applies.
    part_group : FrozenSet[Part]
        The Parts which are covered by the Proposal.
    plans : List[Plan]
        The Plans used in the Proposal.
    input_tensor_configs : Dict[Tensor, TensorConfig]
        The TensorConfigs indexed by Tensor in the Proposal which aren't produced by a Plan.
    cascade_region : MemoryRegion
        The MemoryRegion where cascading buffers should be homed.
    memory_usage : int
        The memory required to execute the Proposal in the cascading MemoryRegion.
    cycles : int
        The estimated cycles taken to execute the Proposal.

    """

    def __init__(
        self,
        graph: CascaderGraph,
        part_group: FrozenSet[Part],
        plans: List[Plan],
        input_tensor_configs: Dict[Tensor, TensorConfig],
        cascade_region: MemoryRegion,
        memory_usage: Dict[MemoryRegion, int],
        cycles: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Proposal,
            graph,
            list(part_group),
            plans,
            input_tensor_configs,
            cascade_region,
            memory_usage,
            cycles,
        )

    @property
    def graph(self) -> CascaderGraph:
        """The CascaderGraph to which the Proposal applies."""
        return self._graph

    @property
    def part_group(self) -> FrozenSet[Part]:
        """The Parts which are covered by the Proposal."""
        return frozenset(self._part_group)

    @property
    def plans(self) -> List[Plan]:
        """The Plans used in the Proposal."""
        return list(self._plans)

    @property
    def input_tensor_configs(self) -> Dict[Tensor, TensorConfig]:
        """The TensorConfigs indexed by Tensor in the Proposal which aren't produced by a Plan."""
        return dict(self._input_tensor_configs)

    @property
    def cascade_region(self) -> MemoryRegion:
        """The MemoryRegion where cascading buffers should be homed."""
        return self._cascade_region

    @property
    def memory_usage(self) -> int:
        """The memory required to execute the Proposal in the cascading MemoryRegion."""
        return int(self._memory_usage)

    @property
    def cycles(self) -> int:
        """The estimated cycles taken to execute the Proposal."""
        return int(self._cycles)
