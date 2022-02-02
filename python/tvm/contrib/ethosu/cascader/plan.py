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
    """
    A class which describes how to schedule a subgraph of Parts together.

    A Plan takes the form of a subgraph of connected Parts (recorded in part_group) with
    TensorConfigs for all of the required Tensors (recorded in tensor_configs). This information
    can be used to produce a Tensor Expression schedule with inter-operator scheduling. A Plan is
    necessarily single-output such that all non-output Parts are 'computed_at'ed the scope of the
    output Part. This is what achieves the technique referred to as 'cascading'. A Plan also has
    an interior memory region which specifies the region of memory into which all the Plans
    intermediate buffers should be allocated.

    Additionally, a Plan contains some other information used during the Plan generation and
    selection algorithms. Both the memory and cycles required to run the Plan are accounted for so
    that Plans can be ranked and Pareto-culled on these metrics. Furthermore, the TensorConfigs
    which are 'open' is recorded indicating that these are valid points to merge with another Plan.
    A Plan can only be turned into a schedule if it has no 'open' TensorConfigs - at which point
    the Plan is said to be 'closed'.

    Attributes
    ----------
    tensor_configs : Dict[Tensor, TensorConfig]
        The TensorConfigs specified by the Plan.
    open_configs : FrozenSet[TensorConfig]
        The TensorConfigs which are 'open' meaning they are a Plan input/output but have
        'interior' state.
    output_config : TensorConfig
        The TensorConfig of the Plan's output tensor.
    part_group : FrozenSet[Part]
        The Parts which are covered by the Plan.
    interior_region : MemoryRegion
        The MemoryRegion in which to store 'interior' Plan buffers.
    memory_usage : int
        The interior memory used by the Plan in bytes.
    cycles : int
        The cycles taken to execute the Plan.

    """

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
        """
        Merge two Plans with share an 'open' TensorConfig.

        The current Plan is referred to as the 'upper Plan' and the other Plan as the 'lower
        Plan'. The 'open' output config of the upper Plan must be an 'open' input config of the
        lower Plan. The Tensor referenced by these configs is the Tensor on which the two Plans
        will be merged. The merge process does the following:

        The tensor config maps will be merged with TensorConfigs from the upper Plan taking
        priority. The open configs will be merged with the TensorConfigs that are being merged
        having been removed. The output config will be that of the lower Plan. The part groups
        will be merged. The interior region is necessarily the same for both the upper and lower
        Plan. The cycles and memory usage will be summed.

        Parameters
        ----------
        other : Plan
            The Plan to merge with.

        Return
        ------
        Plan
            The merged Plan.

        """
        return _ffi_api.PlanMerge(self, other)

    @property
    def tensor_configs(self):
        """The TensorConfigs specified by the Plan."""
        tensor_configs = {}
        for config in self._tensor_configs:
            tensor_configs[config.tensor] = config
        return tensor_configs

    @property
    def open_configs(self):
        """
        The TensorConfigs which are 'open' meaning they are a Plan input/output but have
        'interior' state.
        """
        return frozenset(self._open_configs)

    @property
    def output_config(self):
        """The TensorConfig of the Plan's output tensor."""
        return self._output_config

    @property
    def part_group(self):
        """The Parts which are covered by the Plan."""
        return frozenset(self._part_group)

    @property
    def interior_region(self):
        """The MemoryRegion in which to store 'interior' Plan buffers."""
        return self._interior_region

    @property
    def memory_usage(self):
        """The interior memory used by the Plan in bytes."""
        return self._memory_usage

    @property
    def cycles(self):
        """The cycles taken to execute the Plan."""
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
