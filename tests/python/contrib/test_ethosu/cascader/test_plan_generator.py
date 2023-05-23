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

import tvm.contrib.ethosu.cascader as cs
from .infra import make_simple_home_map, make_options, ethosu_enabled

from tvm.contrib.ethosu.cascader.plan_generator import (
    _generate_output_stripe_configs,
    _generate_single_plans,
    _generate_graph_plans,
)


@pytest.mark.parametrize("stripe_factors", [3, 4, 8, 16, 10])
def test_generate_output_stripe_configs_disable_striping(stripe_factors):
    subgraph = cs.TESubgraph([], None)
    part_1 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([800, 800], "uint8")
    tensor_2 = cs.Tensor([400, 400], "uint8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    assert (
        len(
            _generate_output_stripe_configs(
                part_1, stripe_factors, enable_striping=False, multi_dimensional=False
            )
        )
        == 1
    )


def test_generate_output_stripe_configs_multi_dimensional():
    stripe_factors = 3
    subgraph = cs.TESubgraph([], None)
    part_1 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([800, 800], "uint8")
    tensor_2 = cs.Tensor([400, 400], "uint8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    expected_stripe_configs = {
        cs.StripeConfig([1, 1], [400, 400], [1, 1], [1, 2], [400, 400], [0, 0]),
        cs.StripeConfig([1, 1], [400, 400], [1, 1], [2, 1], [400, 400], [0, 0]),
        cs.StripeConfig([200, 1], [400, 400], [200, 1], [1, 2], [2, 400], [0, 0]),
        cs.StripeConfig([200, 1], [400, 400], [200, 1], [2, 1], [2, 400], [0, 0]),
        cs.StripeConfig([400, 1], [400, 400], [400, 1], [2, 1], [1, 400], [0, 0]),
        cs.StripeConfig([1, 200], [400, 400], [1, 200], [1, 2], [400, 2], [0, 0]),
        cs.StripeConfig([1, 200], [400, 400], [1, 200], [2, 1], [400, 2], [0, 0]),
        cs.StripeConfig([200, 200], [400, 400], [200, 200], [2, 1], [2, 2], [0, 0]),
        cs.StripeConfig([200, 200], [400, 400], [200, 200], [1, 2], [2, 2], [0, 0]),
        cs.StripeConfig([400, 200], [400, 400], [400, 200], [2, 1], [1, 2], [0, 0]),
        cs.StripeConfig([1, 400], [400, 400], [1, 400], [1, 2], [400, 1], [0, 0]),
        cs.StripeConfig([200, 400], [400, 400], [200, 400], [1, 2], [2, 1], [0, 0]),
        cs.StripeConfig([400, 400], [400, 400], [400, 400], [1, 2], [1, 1], [0, 0]),
    }

    output_stripe_configs = _generate_output_stripe_configs(
        part=part_1, stripe_factors=stripe_factors, enable_striping=True, multi_dimensional=True
    )

    assert len(output_stripe_configs) == len(expected_stripe_configs)
    assert set(output_stripe_configs) == expected_stripe_configs


def test_generate_output_stripe_configs_uncascadable_axis():
    stripe_factors = 3
    subgraph = cs.TESubgraph([], None)
    part_1 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[2, 0, 0], [0, 0, 200], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([800, 200], "uint8")
    tensor_2 = cs.Tensor([400, 400], "uint8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    expected_stripe_configs = {
        cs.StripeConfig([1, 400], [400, 400], [1, 400], [1, 2], [400, 1], [0, 0]),
        cs.StripeConfig([200, 400], [400, 400], [200, 400], [1, 2], [2, 1], [0, 0]),
        cs.StripeConfig([400, 400], [400, 400], [400, 400], [1, 2], [1, 1], [0, 0]),
    }

    output_stripe_configs = _generate_output_stripe_configs(
        part=part_1, stripe_factors=stripe_factors, enable_striping=True, multi_dimensional=True
    )

    assert len(output_stripe_configs) == len(expected_stripe_configs)
    assert set(output_stripe_configs) == expected_stripe_configs


def test_generate_output_stripe_configs_single_dimension():
    stripe_factors = 3
    subgraph = cs.TESubgraph([], None)
    part_1 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([800, 800], "uint8")
    tensor_2 = cs.Tensor([400, 400], "uint8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    expected_stripe_configs = {
        cs.StripeConfig([400, 1], [400, 400], [400, 1], [2, 1], [1, 400], [0, 0]),
        cs.StripeConfig([400, 200], [400, 400], [400, 200], [2, 1], [1, 2], [0, 0]),
        cs.StripeConfig([1, 400], [400, 400], [1, 400], [1, 2], [400, 1], [0, 0]),
        cs.StripeConfig([200, 400], [400, 400], [200, 400], [1, 2], [2, 1], [0, 0]),
        cs.StripeConfig([400, 400], [400, 400], [400, 400], [1, 2], [1, 1], [0, 0]),
    }

    output_stripe_configs = _generate_output_stripe_configs(
        part=part_1, stripe_factors=stripe_factors, enable_striping=True, multi_dimensional=False
    )

    assert len(output_stripe_configs) == len(expected_stripe_configs)
    assert set(output_stripe_configs) == expected_stripe_configs


def test_generate_single_plans(SRAM, DRAM):
    subgraph = cs.TESubgraph([], None)
    part_1 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([800, 800], "int8")
    tensor_2 = cs.Tensor([400, 400], "int8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    home_map = {
        tensor_1: [SRAM, DRAM],
        tensor_2: [SRAM],
    }
    options = make_options(cascade_region=SRAM, stripe_factors=1)
    output_stripe_configs = _generate_output_stripe_configs(
        part_1,
        options.stripe_factors,
        enable_striping=True,
        multi_dimensional=True,
    )
    plans = _generate_single_plans(part_1, output_stripe_configs, home_map, options)
    for plan in plans:
        assert plan.interior_region == SRAM
        assert plan.part_group == frozenset([part_1])
        assert set(plan.tensor_configs.keys()) == set([tensor_1, tensor_2])
        for open_config in plan.open_configs:
            assert open_config.state == cs.TensorConfigState.INTERIOR


def test_generate_graph_plans(SRAM, DRAM):
    num_part_groups = 3
    stripe_factors = 4
    max_plan_size = 10
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
    tensor_1 = cs.Tensor([10, 10], "int8")
    tensor_2 = cs.Tensor([9, 9], "int8")
    tensor_3 = cs.Tensor([10, 10], "int8")
    tensor_4 = cs.Tensor([10, 10], "int8")

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

    graph = cs.CascaderGraph([tensor_1, tensor_2], [tensor_4])
    home_map = {
        tensor_1: [SRAM, DRAM],
        tensor_2: [SRAM],
        tensor_3: [SRAM],
        tensor_4: [SRAM, DRAM],
    }

    options = make_options(
        cascade_region=SRAM,
        stripe_factors=stripe_factors,
        max_plan_size=max_plan_size,
    )
    closed_plans = _generate_graph_plans(graph, home_map, options)

    assert len(closed_plans) == num_part_groups


if ethosu_enabled:

    def test_plan_generator_two_conv2d(FLASH, SRAM, TwoConv2DGraph):
        num_part_groups = 3
        graph = TwoConv2DGraph
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            stripe_factors=4,
            max_plan_size=10,
        )

        closed_plans = _generate_graph_plans(graph, home_map, options)

        assert len(closed_plans) == num_part_groups

    def test_plan_generator_two_conv2d_with_slice(FLASH, SRAM, TwoConv2DWithSliceGraph):
        num_part_groups = 4  # Note this is not 6 because 'slice' has an opaque Propagator
        graph = TwoConv2DWithSliceGraph
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            stripe_factors=4,
            max_plan_size=10,
        )

        closed_plans = _generate_graph_plans(graph, home_map, options)

        assert len(closed_plans) == num_part_groups


if __name__ == "__main__":
    tvm.testing.main()
