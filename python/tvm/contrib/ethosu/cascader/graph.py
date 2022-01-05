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
"""Graph objects to define compute graphs for the NPU cascader."""
from typing import List
from collections import namedtuple
import tvm._ffi

from tvm.runtime import Object

from .stripe_config import StripeConfig
from . import _ffi_api


TESubgraph = namedtuple("TESubgraph", ["input_tensors", "output_tensor"])


@tvm._ffi.register_object("contrib.ethosu.cascader.PerformanceInfo")
class PerformanceInfo(Object):
    """PerformanceInfo class"""

    @property
    def compute_cycles(self):
        return self._compute_cycles

    @property
    def read_bytes(self):
        return list(self._read_bytes)

    @property
    def write_bytes(self):
        return self._write_bytes


@tvm._ffi.register_object("contrib.ethosu.cascader.Tensor")
class Tensor(Object):
    """Tensor class"""

    def __init__(self, shape, dtype, is_constant=False, compression_ratio=1):
        self.__init_handle_by_constructor__(
            _ffi_api.Tensor, shape, dtype, is_constant, compression_ratio
        )

    def add_producer(self, part):
        _ffi_api.TensorAddProducer(self, part)

    def add_consumer(self, part):
        _ffi_api.TensorAddConsumer(self, part)

    @property
    def producers(self):
        return list(self._producers)

    @property
    def consumers(self):
        return list(self._consumers)

    @property
    def shape(self):
        return list(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_constant(self):
        return self._is_constant

    @property
    def compression_ratio(self):
        return self._compression_ratio

    @property
    def size(self):
        return self._size


class Part(Object):
    """Part base class"""

    def set_input(self, index: int, tensor: Tensor):
        _ffi_api.PartSetInput(self, index, tensor)

    def set_output(self, tensor: Tensor):
        _ffi_api.PartSetOutput(self, tensor)

    def calculate_input_stripe_configs(
        self, output_stripe_config: StripeConfig
    ) -> List[StripeConfig]:
        return list(_ffi_api.PartCalculateInputStripeConfigs(self, output_stripe_config))

    def get_stripe_align_hint(self) -> List[int]:
        return list(_ffi_api.PartGetStripeAlignHint(self))

    def get_performance_info(
        self, stripe_config: StripeConfig, is_rolling: bool
    ) -> PerformanceInfo:
        return _ffi_api.PartGetPerformanceInfo(self, stripe_config, is_rolling)

    @property
    def input_tensors(self):
        return list(self._input_tensors)

    @property
    def output_tensor(self):
        return self._output_tensor

    @property
    def propagators(self):
        return list(self._propagators)

    @property
    def in_line(self):
        return self._in_line

    @property
    def subgraph(self):
        return TESubgraph(list(self._te_input_tensors), self._te_output_tensor)


@tvm._ffi.register_object("contrib.ethosu.cascader.CascaderGraph")
class CascaderGraph(Object):
    """A class to describe a graph of Parts and Tensors used by the cascader.

    This class describes a graph consisting of two object types: Tensors and Parts.
    It defines a topological ordering on the graph such that each Part and Tensor has a
    position in the ordering. This ordering is used by the Plan and Proposal generation
    algorithms. It is also the ordering the Parts are expected to be executed in.

    In addition to defining an ordering, the Parts and Tensors are also all given unique
    IDs which they can be referred to by."""

    def __init__(self, input_tensors: List[Tensor], output_tensors: List[Tensor]):
        self.__init_handle_by_constructor__(_ffi_api.CascaderGraph, input_tensors, output_tensors)

    def get_part_id(self, part: Part) -> int:
        return _ffi_api.CascaderGraphGetPartID(self, part)

    def get_tensor_id(self, tensor: Tensor) -> int:
        return _ffi_api.CascaderGraphGetTensorID(self, tensor)

    @property
    def input_tensors(self):
        return list(self._input_tensors)

    @property
    def output_tensors(self):
        return list(self._output_tensors)

    @property
    def tensor_order(self):
        return list(self._tensor_order)

    @property
    def part_order(self):
        return list(self._part_order)
