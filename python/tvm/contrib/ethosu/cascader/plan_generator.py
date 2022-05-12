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
"""Algorithms to generate Plans for a CascaderGraph."""
from typing import List, Dict

from tvm.contrib.ethosu.cascader.tensor_config import MemoryRegion

from . import _ffi_api
from .cascader_options import CascaderOptions
from .plan import Plan
from .stripe_config import StripeConfig
from .graph import CascaderGraph, Part, Tensor


def _generate_output_stripe_configs(
    part: Part, stripe_factors: int, enable_striping: bool, multi_dimensional: bool
) -> List[StripeConfig]:
    return list(
        _ffi_api.GenerateOutputStripeConfigs(
            part, stripe_factors, enable_striping, multi_dimensional
        )
    )


def _generate_single_plans(
    part: Part,
    output_stripe_configs: List[StripeConfig],
    home_map: Dict[Tensor, List[MemoryRegion]],
    cascade_region: MemoryRegion,
) -> List[Plan]:
    return list(_ffi_api.GenerateSinglePlans(part, output_stripe_configs, home_map, cascade_region))


def _generate_graph_plans(
    graph: CascaderGraph,
    home_map: Dict[Tensor, List[MemoryRegion]],
    options: CascaderOptions,
):
    return _ffi_api.GenerateGraphPlans(
        graph,
        home_map,
        options,
    )
