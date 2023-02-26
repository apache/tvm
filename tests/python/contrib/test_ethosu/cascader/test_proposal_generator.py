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
from tvm.contrib.ethosu.cascader.proposal_generator import generate_proposals

from .infra import make_simple_home_map, make_options, ethosu_enabled


if ethosu_enabled:

    def test_generate_proposals(FLASH, SRAM, TwoConv2DGraph):
        graph = TwoConv2DGraph
        min_sram = 3700
        max_sram = 11700
        input_configs = 1
        parts = 2
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=32,
            stripe_factors=4,
            max_plan_size=10,
        )

        proposals = generate_proposals(graph, home_map, options)

        for proposal in proposals:
            assert 0 < len(proposal.plans) <= parts
            assert len(proposal.input_tensor_configs) == input_configs
            assert len(proposal.part_group) == parts
            assert min_sram < proposal.memory_usage < max_sram
            assert proposal.cycles > 0

    def test_generate_proposals_binary(FLASH, SRAM, BinaryGraph):
        graph = BinaryGraph
        input_configs = 2
        parts = 3
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=32,
            stripe_factors=4,
            max_plan_size=10,
        )

        proposals = generate_proposals(graph, home_map, options)

        for proposal in proposals:
            assert 0 < len(proposal.plans) <= parts
            assert len(proposal.input_tensor_configs) == input_configs
            assert len(proposal.part_group) == parts
            assert proposal.cycles > 0

    def test_generate_proposals_mobilenetv1_start(FLASH, SRAM, MobileNetv1StartGraph):
        graph = MobileNetv1StartGraph
        min_sram = 200000
        max_sram = 1300000
        input_configs = 1
        parts = 8
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=32,
            stripe_factors=5,
            max_plan_size=10,
        )

        proposals = generate_proposals(graph, home_map, options)

        for proposal in proposals:
            assert 0 < len(proposal.plans) <= parts
            assert len(proposal.input_tensor_configs) == input_configs
            assert len(proposal.part_group) == parts
            assert min_sram < proposal.memory_usage < max_sram
            assert proposal.cycles > 0

    def test_generate_proposals_mobilenetv1(FLASH, SRAM, MobileNetv1Graph):
        graph = MobileNetv1Graph
        min_sram = 200000
        max_sram = 1300000
        input_configs = 1
        parts = 27
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=32,
            stripe_factors=5,
            max_plan_size=10,
        )

        proposals = generate_proposals(graph, home_map, options)

        for proposal in proposals:
            assert 0 < len(proposal.plans) <= parts
            assert len(proposal.input_tensor_configs) == input_configs
            assert len(proposal.part_group) == parts
            assert min_sram < proposal.memory_usage < max_sram
            assert proposal.cycles > 0

    def test_generate_proposals_mobilenetv2diamond(FLASH, SRAM, MobileNetv2DiamondGraph):
        graph = MobileNetv2DiamondGraph
        min_sram = 370000
        max_sram = 990000
        input_configs = 1
        parts = 5
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=64,
            stripe_factors=5,
            max_plan_size=10,
        )

        proposals = generate_proposals(graph, home_map, options)

        for proposal in proposals:
            assert 0 < len(proposal.plans) <= parts
            assert len(proposal.input_tensor_configs) == input_configs
            assert len(proposal.part_group) == parts
            assert min_sram < proposal.memory_usage < max_sram
            assert proposal.cycles > 0

    def test_generate_proposals_mobilenetv1_disable_striping(FLASH, SRAM, MobileNetv1Graph):
        graph = MobileNetv1Graph
        home_map = make_simple_home_map(graph, SRAM, FLASH)
        options = make_options(
            cascade_region=SRAM,
            max_proposals=32,
            stripe_factors=5,
            max_plan_size=10,
            enable_striping=False,
        )

        proposals = generate_proposals(graph, home_map, options)
        assert len(proposals) == 1
        proposal = proposals[0]
        for plan in proposal.plans:
            for stripe_config in plan.output_config.stripe_configs:
                for shape_dim, stride_dim in list(zip(stripe_config.shape, stripe_config.strides)):
                    # The striding and shape sizes in each dimension should be the same
                    # if striping is disabled
                    assert int(shape_dim) == int(stride_dim)


if __name__ == "__main__":
    tvm.testing.main()
