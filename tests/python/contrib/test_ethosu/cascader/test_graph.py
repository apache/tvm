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
import pytest

pytest.importorskip("ethosu.vela")

import tvm.contrib.ethosu.cascader as cs


def test_tensor():
    shape = [1, 2, 3]
    dtype = "uint8"
    is_constant = True
    compression_ratio = 0.5
    size = 6
    tensor = cs.Tensor(shape, dtype, is_constant, compression_ratio)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.is_constant == is_constant
    assert tensor.compression_ratio == compression_ratio
    assert tensor.size == size


def test_inline_part():
    subgraph = cs.TESubgraph([], None)
    part = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    output_stripe_config = cs.StripeConfig([2, 4], [8, 8], [2, 4], [1, 2], [4, 2], [0, 0])
    input_stripe_config = cs.StripeConfig([4, 2], [8, 8], [4, 2], [2, 1], [2, 4], [0, 0])

    assert part.input_tensors == [None]
    assert part.output_tensor == None
    assert len(part.propagators) == 1
    assert part.in_line == True
    assert part.get_stripe_align_hint() == [1, 1]
    performance_info = part.get_performance_info(output_stripe_config, cs.BufferMode.RECOMPUTE)
    assert performance_info.compute_cycles == 0
    assert performance_info.read_bytes == [0]
    assert performance_info.write_bytes == 0
    input_stripe_configs = part.calculate_input_stripe_configs(output_stripe_config)
    assert len(input_stripe_configs) == 1
    assert input_stripe_configs[0] == input_stripe_config


def test_small_graph():
    subgraph = cs.TESubgraph([], None)
    part_a = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0, 0],
            ),
            cs.Propagator(
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                [-1, -1],
            ),
        ],
    )
    part_b = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([10, 10], "uint8")
    tensor_2 = cs.Tensor([9, 9], "uint8")
    tensor_3 = cs.Tensor([10, 10], "uint8")
    tensor_4 = cs.Tensor([10, 10], "uint8")

    part_a.set_input(0, tensor_1)
    part_a.set_input(1, tensor_2)
    part_a.set_output(tensor_3)
    tensor_1.add_consumer(part_a)
    tensor_2.add_consumer(part_a)
    tensor_3.add_producer(part_a)
    part_b.set_input(0, tensor_3)
    part_b.set_output(tensor_4)
    tensor_3.add_consumer(part_b)
    tensor_4.add_producer(part_b)

    assert part_a.input_tensors == [tensor_1, tensor_2]
    assert part_a.output_tensor == tensor_3
    assert part_b.input_tensors == [tensor_3]
    assert part_b.output_tensor == tensor_4

    assert tensor_1.producers == []
    assert tensor_1.consumers == [part_a]
    assert tensor_2.producers == []
    assert tensor_2.consumers == [part_a]
    assert tensor_3.producers == [part_a]
    assert tensor_3.consumers == [part_b]
    assert tensor_4.producers == [part_b]
    assert tensor_4.consumers == []

    graph = cs.CascaderGraph([tensor_1, tensor_2], [tensor_4])
    assert graph.input_tensors == [tensor_1, tensor_2]
    assert graph.output_tensors == [tensor_4]
    assert graph.part_order == [part_b, part_a]
    for i, part in enumerate(graph.part_order):
        assert graph.get_part_id(part) == i


def test_create_cascader_graph(TwoConv2DWithSliceTE):
    _, te_graph, const_dict = TwoConv2DWithSliceTE
    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    graph = cs.create_cascader_graph(te_graph, const_dict, device_config)

    output_tensor = graph.output_tensors[0]
    assert output_tensor.shape == [1, 6, 1, 6, 16]
    assert len(output_tensor.producers) == 1
    assert not output_tensor.is_constant

    conv2_part = output_tensor.producers[0]
    assert isinstance(conv2_part, cs.EthosuPart)
    assert len(conv2_part.input_tensors) == 3

    assert conv2_part.input_tensors[0].shape == [1, 6, 6, 64]
    assert len(conv2_part.input_tensors[0].producers) == 1
    assert not conv2_part.input_tensors[0].is_constant

    assert conv2_part.input_tensors[1].shape == [16, 3, 3, 64]
    assert len(conv2_part.input_tensors[1].producers) == 0
    assert conv2_part.input_tensors[1].is_constant

    assert conv2_part.input_tensors[2].shape == [16, 10]
    assert len(conv2_part.input_tensors[2].producers) == 0
    assert conv2_part.input_tensors[2].is_constant

    slice_part = conv2_part.input_tensors[0].producers[0]
    assert isinstance(slice_part, cs.InlinePart)
    assert len(slice_part.input_tensors) == 1

    assert slice_part.input_tensors[0].shape == [1, 12, 12, 64]
    assert len(slice_part.input_tensors[0].producers) == 1
    assert not slice_part.input_tensors[0].is_constant

    conv1_part = slice_part.input_tensors[0].producers[0]
    assert isinstance(conv1_part, cs.EthosuPart)
    assert len(conv1_part.input_tensors) == 3

    assert conv1_part.input_tensors[0].shape == [1, 12, 12, 8]
    assert len(conv1_part.input_tensors[0].producers) == 0
    assert not conv1_part.input_tensors[0].is_constant

    assert conv1_part.input_tensors[1].shape == [64, 1, 1, 8]
    assert len(conv1_part.input_tensors[1].producers) == 0
    assert conv1_part.input_tensors[1].is_constant

    assert conv1_part.input_tensors[2].shape == [64, 10]
    assert len(conv1_part.input_tensors[2].producers) == 0
    assert conv1_part.input_tensors[2].is_constant


def test_create_diamond_graph(MobileNetv2DiamondTE):
    _, te_graph, const_dict = MobileNetv2DiamondTE
    device_config = cs.EthosuDeviceConfig("ethos-u55-256")
    graph = cs.create_cascader_graph(te_graph, const_dict, device_config)

    output_tensor = graph.output_tensors[0]
    assert output_tensor.shape == [1, 56, 56, 24]
    assert len(output_tensor.producers) == 1
    assert not output_tensor.is_constant

    add1_part = output_tensor.producers[0]
    assert isinstance(add1_part, cs.EthosuPart)
    assert len(add1_part.input_tensors) == 2
    assert graph.get_part_id(add1_part) == 0

    assert add1_part.input_tensors[0].shape == [1, 56, 56, 24]
    assert len(add1_part.input_tensors[0].producers) == 1
    assert not add1_part.input_tensors[0].is_constant

    assert add1_part.input_tensors[1].shape == [1, 56, 56, 24]
    assert len(add1_part.input_tensors[0].producers) == 1
    assert not add1_part.input_tensors[0].is_constant


if __name__ == "__main__":
    tvm.testing.main()
