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
from typing import List, Dict
from enum import IntEnum
from collections import namedtuple
import numpy as np

import tvm._ffi
from tvm import te
from tvm.runtime import Object

from .stripe_config import StripeConfig
from .device_config import EthosuDeviceConfig
from . import _ffi_api


# A global store to register matching functions
REGISTERED_MATCHERS = []


TESubgraph = namedtuple("TESubgraph", ["input_tensors", "output_tensor"])


class BufferMode(IntEnum):
    RECOMPUTE = 0
    ROLLING = 1


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

    @property
    def block_config(self):
        return self._block_config


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
        self, stripe_config: StripeConfig, buffer_mode: BufferMode
    ) -> PerformanceInfo:
        return _ffi_api.PartGetPerformanceInfo(self, stripe_config, buffer_mode)

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


def register_matcher(matcher):
    """Register a match function to the frontend.

    A match function takes a te.Tensor and checks whether it matches
    a known operator/operator sequence. If it does, it returns a Part
    which models the behaviour of that operator sequence. Otherwise,
    it returns None.
    """
    REGISTERED_MATCHERS.append(matcher)
    return matcher


def create_cascader_graph(
    te_graph: TESubgraph, const_dict: Dict[int, np.ndarray], device_config: EthosuDeviceConfig
) -> CascaderGraph:
    """Create a CascaderGraph from a Tensor Expression graph and constant dictionary.

    Parameters
    ----------
    te_graph : TESubgraph
        The Tensor Expression graph.
    const_dict : Dict[int, np.ndarray]
        The constant dictionary.
    device_config : EthosuDeviceConfig
        Target device configuration.

    Returns
    -------
    CascaderGraph
        The CascaderGraph.
    """
    tensor_map = {}

    def _visit_tensor(tensor):
        if tensor not in tensor_map:
            is_const = False
            # Logic to determine if the tensor is constant
            if tensor in list(te_graph.inputs):
                i = list(te_graph.inputs).index(tensor)
                if i in const_dict:
                    is_const = True

            # TODO(@mbaret): Calculate the compression ratio
            plan_tensor = Tensor(
                tensor.shape,
                tensor.dtype,
                is_constant=is_const,
            )
            tensor_map[tensor] = plan_tensor
            if isinstance(tensor.op, te.PlaceholderOp) or tensor in te_graph.inputs:
                return

            input_tensors = []
            # Check whether any of the registered matchers match the current tensor
            for matcher in REGISTERED_MATCHERS:
                part = matcher(tensor, device_config)
                if part:
                    input_tensors = part.subgraph.input_tensors
                    break

            assert part is not None, f"The tensor {tensor} doesn't match any part."
            part.set_output(plan_tensor)
            plan_tensor.add_producer(part)
            for i, input_tensor in enumerate(input_tensors):
                _visit_tensor(input_tensor)
                part.set_input(i, tensor_map[input_tensor])
                tensor_map[input_tensor].add_consumer(part)

    for output in te_graph.outputs:
        _visit_tensor(output)

    input_tensors = []
    for t in te_graph.inputs:
        # This is needed because sometimes there are orphaned constants
        if t in tensor_map:
            input_tensors.append(tensor_map[t])

    output_tensors = [tensor_map[t] for t in te_graph.outputs]
    return CascaderGraph(input_tensors, output_tensors)
