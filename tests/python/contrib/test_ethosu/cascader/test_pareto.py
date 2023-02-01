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
from tvm.tir import IntImm
from tvm.contrib.ethosu.cascader.pareto import (
    _get_pareto_frontier,
    _thin_vector,
    _pareto_cull_plans,
)
from tvm.contrib.ethosu.cascader import (
    Plan,
    StripeConfig,
    TensorConfig,
    TensorConfigState,
    BufferMode,
    Tensor,
)

import pytest
import numpy as np


def _ref_get_pareto_frontier(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def _ref_thin_vector(vec, max_size):
    if max_size < 1:
        return []
    if len(vec) <= max_size or len(vec) == 0:
        return vec
    if max_size == 1:
        return [vec[0]]
    samples = np.linspace(0, len(vec), max_size - 1, endpoint=False).astype(int)
    samples = np.append(samples, len(vec) - 1)
    return vec[samples]


def _ref_pareto_cull_plans(plans, points):
    if len(plans) <= points:
        return plans
    plans = np.array(sorted(plans, key=lambda x: x.memory_usage))
    costs = []
    for plan in plans:
        costs.append(np.array([plan.memory_usage, plan.cycles]))
    is_efficient = _ref_get_pareto_frontier(np.array(costs))
    culled_plans = plans[is_efficient]
    thinned_plans = (
        culled_plans
        if len(culled_plans) <= points
        else _ref_thin_vector(np.array(culled_plans), points)
    )
    return thinned_plans


@pytest.mark.parametrize("num_costs", [1, 10, 30, 100, 300, 1000])
def test_get_pareto_frontier(num_costs):
    cost_low = 1
    cost_high = 100
    dims = 2
    costs = []
    for i in range(num_costs):
        costs.append(list(np.random.randint(cost_low, cost_high, size=(dims,))))
    reference = list(_ref_get_pareto_frontier(np.array(costs)))
    result = _get_pareto_frontier(costs)
    assert result == reference


@pytest.mark.parametrize("vec_length", [0, 1, 10, 25, 100])
@pytest.mark.parametrize("max_size", [0, 1, 2, 5, 11, 51])
def test_thin_vector(vec_length, max_size):
    def _make_vector(length):
        vector = []
        for i in range(length):
            obj = IntImm("int32", i)
            vector.append(obj)

        return vector

    vector = _make_vector(vec_length)
    reference = list(_ref_thin_vector(np.array(vector), max_size))
    result = _thin_vector(vector, max_size)
    assert result == reference


@pytest.mark.parametrize("num_plans", [0, 1, 10, 25, 100])
@pytest.mark.parametrize("max_plans", [0, 1, 2, 5, 11, 51])
def test_pareto_cull_plans(num_plans, max_plans, SRAM):
    memory_usage_low = 1
    memory_usage_high = 1000
    cycles_low = 100
    cycles_high = 10000

    def _make_plan(memory_usage, cycles):
        output_config = TensorConfig(
            tensor=Tensor([1], "int8"),
            home_region=SRAM,
            state=TensorConfigState.BOUNDARY,
            buffer_mode=BufferMode.RECOMPUTE,
            stripe_configs=[StripeConfig([1], [1], [1], [1], [1], [0])],
        )
        return Plan(
            tensor_configs={},
            open_configs=[],
            output_config=output_config,
            part_group=[],
            interior_region=SRAM,
            memory_usage=memory_usage,
            cycles=cycles,
        )

    def _make_plans(num):
        plans = []
        for _ in range(num):
            memory_usage = np.random.randint(memory_usage_low, memory_usage_high)
            cycles = np.random.randint(cycles_low, cycles_high)
            plan = _make_plan(memory_usage, cycles)
            plans.append(plan)

        return plans

    plans = _make_plans(num_plans)
    reference = list(_ref_pareto_cull_plans(plans, max_plans))
    result = _pareto_cull_plans(plans, max_plans, False)
    assert result == reference


if __name__ == "__main__":
    tvm.testing.main()
