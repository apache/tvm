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
"""Algorithms to generate Proposals for a Graph."""
from typing import List, Dict, FrozenSet

from . import _ffi_api
from .cascader_options import CascaderOptions
from .plan import Plan
from .proposal import Proposal
from .graph import CascaderGraph, Part


def generate_proposals(
    graph: CascaderGraph,
    home_map: Dict[FrozenSet[Part], List[Plan]],
    options: CascaderOptions,
) -> List[Proposal]:
    """Generate Pareto optimal Proposals for a CascaderGraph.

    This algorithm takes a top-down dynamic programming approach to determining how
    to optimally combine Plans into Proposals.

    Parameters
    ----------
    graph : CascaderGraph
        The CascaderGraph to generate Proposals for.
    home_map : Dict[FrozenSet[Part], List[Plan]]
        The Tensor homing map defining valid memory homes for Tensors.
    options : CascaderOptions
        The configuration options with which to run the generator.

    Returns
    ------
    List[Proposal]
        A list of Pareto optimal Proposals.

    """
    return list(
        _ffi_api.GenerateProposals(
            graph,
            home_map,
            options,
        )
    )
