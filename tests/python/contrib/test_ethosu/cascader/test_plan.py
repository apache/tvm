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
import tvm.contrib.ethosu.cascader as cs

import pytest


def test_plan(DRAM, SRAM):
    subgraph = cs.TESubgraph([], None)
    part = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0, 0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([10, 10], "uint8")
    tensor_2 = cs.Tensor([10, 10], "uint8")

    part.set_input(0, tensor_1)
    part.set_output(tensor_2)
    tensor_1.add_consumer(part)
    tensor_2.add_producer(part)

    output_stripe_config = cs.StripeConfig(
        shape=[5, 5],
        extent=[10, 10],
        strides=[5, 5],
        order=[1, 2],
        stripes=[2, 2],
        offset=[0, 0],
    )
    tensor_config_out = cs.TensorConfig(
        tensor=tensor_2,
        home_region=DRAM,
        state=cs.TensorConfigState.BOUNDARY,
        buffer_mode=cs.BufferMode.RECOMPUTE,
        stripe_configs=[output_stripe_config],
        copy_tensor=False,
    )
    input_stripe_config = part.calculate_input_stripe_configs(output_stripe_config)[0]
    tensor_config_in = cs.TensorConfig(
        tensor=tensor_1,
        home_region=DRAM,
        state=cs.TensorConfigState.INTERIOR,
        buffer_mode=cs.BufferMode.ROLLING,
        stripe_configs=[input_stripe_config],
        copy_tensor=False,
    )
    tensor_configs = {tensor_1: tensor_config_in, tensor_2: tensor_config_out}
    open_configs = frozenset([tensor_config_in])
    part_group = frozenset([part])
    interior_region = SRAM
    memory_usage = 100
    cycles = 20
    plan = cs.Plan(
        tensor_configs=tensor_configs,
        open_configs=open_configs,
        output_config=tensor_config_out,
        part_group=part_group,
        interior_region=interior_region,
        memory_usage=memory_usage,
        cycles=cycles,
    )

    assert plan.tensor_configs == tensor_configs
    assert plan.open_configs == open_configs
    assert plan.output_config == tensor_config_out
    assert plan.part_group == part_group
    assert plan.interior_region == interior_region
    assert plan.memory_usage == memory_usage
    assert plan.cycles == cycles


def test_plan_merge(DRAM, SRAM):
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
    part_2 = cs.InlinePart(
        subgraph,
        [
            cs.Propagator(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0, 0],
            ),
            cs.Propagator(
                [[0, 0, 6], [0, 0, 6], [0, 0, 1]],
                [0, 0],
            ),
            cs.Propagator(
                [[1, 0], [0, 1]],
                [0],
            ),
        ],
    )
    tensor_1 = cs.Tensor([20, 20], "uint8")
    tensor_2 = cs.Tensor([10, 10], "uint8")
    tensor_3 = cs.Tensor([6, 6], "uint8")
    tensor_4 = cs.Tensor([10], "uint8")
    tensor_5 = cs.Tensor([10, 10], "uint8")

    part_1.set_input(0, tensor_1)
    part_1.set_output(tensor_2)
    tensor_1.add_consumer(part_1)
    tensor_2.add_producer(part_1)

    part_2.set_input(0, tensor_2)
    part_2.set_input(1, tensor_3)
    part_2.set_input(2, tensor_4)
    part_2.set_output(tensor_5)
    tensor_2.add_consumer(part_2)
    tensor_3.add_consumer(part_2)
    tensor_4.add_consumer(part_2)
    tensor_5.add_producer(part_2)

    output_stripe_config = cs.StripeConfig(
        shape=[5, 5],
        extent=[10, 10],
        strides=[5, 5],
        order=[1, 2],
        stripes=[2, 2],
        offset=[0, 0],
    )
    tensor_config_5 = cs.TensorConfig(
        tensor=tensor_5,
        home_region=DRAM,
        state=cs.TensorConfigState.BOUNDARY,
        buffer_mode=cs.BufferMode.RECOMPUTE,
        stripe_configs=[output_stripe_config],
        copy_tensor=False,
    )
    input_stripe_configs = part_2.calculate_input_stripe_configs(output_stripe_config)
    tensor_config_4 = cs.TensorConfig(
        tensor=tensor_4,
        home_region=DRAM,
        state=cs.TensorConfigState.BOUNDARY,
        buffer_mode=cs.BufferMode.RECOMPUTE,
        stripe_configs=[input_stripe_configs[2]],
        copy_tensor=False,
    )
    tensor_config_3 = cs.TensorConfig(
        tensor=tensor_3,
        home_region=SRAM,
        state=cs.TensorConfigState.INTERIOR,
        buffer_mode=cs.BufferMode.RECOMPUTE,
        stripe_configs=[input_stripe_configs[1]],
        copy_tensor=False,
    )
    tensor_config_2 = cs.TensorConfig(
        tensor=tensor_2,
        home_region=SRAM,
        state=cs.TensorConfigState.INTERIOR,
        buffer_mode=cs.BufferMode.ROLLING,
        stripe_configs=[input_stripe_configs[0]],
        copy_tensor=False,
    )
    input_stripe_config = part_1.calculate_input_stripe_configs(input_stripe_configs[0])[0]
    tensor_config_1 = cs.TensorConfig(
        tensor=tensor_1,
        home_region=DRAM,
        state=cs.TensorConfigState.BOUNDARY,
        buffer_mode=cs.BufferMode.ROLLING,
        stripe_configs=[input_stripe_config],
        copy_tensor=False,
    )
    tensor_configs = {tensor_1: tensor_config_1, tensor_2: tensor_config_2}
    open_configs = frozenset([tensor_config_2])
    part_group = frozenset([part_1])
    interior_region = SRAM
    memory_usage = 100
    cycles = 20
    plan_1 = cs.Plan(
        tensor_configs=tensor_configs,
        open_configs=open_configs,
        output_config=tensor_config_2,
        part_group=part_group,
        interior_region=interior_region,
        memory_usage=memory_usage,
        cycles=cycles,
    )

    tensor_configs = {
        tensor_2: tensor_config_2,
        tensor_3: tensor_config_3,
        tensor_4: tensor_config_4,
        tensor_5: tensor_config_5,
    }
    open_configs = frozenset([tensor_config_2, tensor_config_3])
    part_group = frozenset([part_2])
    interior_region = SRAM
    memory_usage = 200
    cycles = 30
    plan_2 = cs.Plan(
        tensor_configs=tensor_configs,
        open_configs=open_configs,
        output_config=tensor_config_5,
        part_group=part_group,
        interior_region=interior_region,
        memory_usage=memory_usage,
        cycles=cycles,
    )

    merged_plan = plan_1.merge(plan_2)

    assert merged_plan.tensor_configs == {
        tensor_1: tensor_config_1,
        tensor_2: tensor_config_2,
        tensor_3: tensor_config_3,
        tensor_4: tensor_config_4,
        tensor_5: tensor_config_5,
    }
    assert merged_plan.open_configs == frozenset([tensor_config_3])
    assert merged_plan.output_config == tensor_config_5
    assert merged_plan.part_group == frozenset([part_1, part_2])
    assert merged_plan.interior_region == interior_region
    assert merged_plan.memory_usage == plan_1.memory_usage + plan_2.memory_usage
    assert merged_plan.cycles == plan_1.cycles + plan_2.cycles


if __name__ == "__main__":
    tvm.testing.main()
