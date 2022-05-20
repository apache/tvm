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
"""Pareto optimisation functions for the NPU cascader."""
from typing import List

from tvm import Object

from . import _ffi_api
from .plan import Plan


def _get_pareto_frontier(costs: List[List[float]]) -> List[bool]:
    for i, cost in enumerate(costs):
        for j, value in enumerate(cost):
            costs[i][j] = float(value)

    return [bool(v) for v in _ffi_api.GetParetoFrontier(costs)]


def _thin_vector(vec: List[Object], max_size: int) -> List[Object]:
    return list(_ffi_api.ThinVector(vec, max_size))


def _pareto_cull_plans(
    plans: List[Plan], max_plans: int, disable_pareto_metric: bool
) -> List[Plan]:
    return list(_ffi_api.ParetoCullPlans(plans, max_plans, disable_pareto_metric))
