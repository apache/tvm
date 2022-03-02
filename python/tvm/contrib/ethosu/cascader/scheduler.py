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
# pylint: disable=invalid-name
"""Scheduler for cascader which converts Proposals into Schedules."""
from typing import Tuple, List, Dict, DefaultDict
from collections import defaultdict
import numpy as np

from tvm import te
from tvm import tir
from .cascader_options import CascaderOptions
from .graph import CascaderGraph, Part, Tensor, TESubgraph
from .tensor_config import MemoryRegion
from .proposal import Proposal
from .proposal_generator import generate_proposals
from .graph import create_cascader_graph
from .device_config import EthosuDeviceConfig


def tile_nd(
    sch: te.Schedule, tensor: te.Tensor, tile: Tuple[int, ...]
) -> Tuple[List[tir.IterVar], List[tir.IterVar]]:
    """Scheduling utility to perform N-dimensional tiling.

    Parameters
    ----------
    sch : te.Schedule
        The schedule to apply the tiling to.
    tensor : te.Tensor
        The tensor to apply the tiling to.
    tile : Tuple[int, ...]
        The N-dimensional tile size.

    Returns
    -------
    outer_indices : List[tir.IterVar]
        The outer iteration variables.
    inner_indices : List[tir.IterVar]
        The inner iteration variables.

    """
    outer_indices = []
    inner_indices = []
    for i, size in enumerate(tile):
        outer, inner = sch[tensor].split(tensor.op.axis[i], size)
        outer_indices.append(outer)
        inner_indices.append(inner)

    sch[tensor].reorder(*outer_indices, *inner_indices)
    return outer_indices, inner_indices


def stripe_part(
    part: Part, stripe_shape: Tuple[int, ...], sch: te.Schedule
) -> Tuple[te.Stage, tir.IterVar]:
    """Apply a striping schedule to the TE subgraph represented by a Part."""
    te_subgraph = part.subgraph
    te_output_tensor = te_subgraph.output_tensor
    outer_indices, _ = tile_nd(sch, te_output_tensor, stripe_shape)
    g = sch.create_group(
        outputs=te_output_tensor.op.input_tensors,
        inputs=te_subgraph.input_tensors,
        include_inputs=False,
    )
    g.compute_at(sch[te_output_tensor], outer_indices[-1])
    for ax in outer_indices:
        sch[te_output_tensor].unroll(ax)

    return sch[te_output_tensor], outer_indices[-1]


def cascade_part(
    part: Part, stripe_stage: te.Stage, stripe_axis: tir.IterVar, sch: te.Schedule
) -> None:
    """Schedule a Part into a cascade indicated by a stripe Stage."""
    te_subgraph = part.subgraph
    g = sch.create_group(
        outputs=te_subgraph.output_tensor, inputs=te_subgraph.input_tensors, include_inputs=False
    )
    g.compute_at(stripe_stage, stripe_axis)


def update_readers(part: Part, readers: DefaultDict[te.Tensor, List[te.Tensor]]) -> None:
    """
    Update a dictionary which stores the te.Tensors that need to be read in
    order to produce a given te.Tensor.
    """
    visited = set()

    def _visit(tensor):
        if tensor not in visited and tensor not in part.subgraph.input_tensors:
            visited.add(tensor)
            for input_tensor in tensor.op.input_tensors:
                readers[input_tensor].append(tensor)
                _visit(input_tensor)

    _visit(part.subgraph.output_tensor)


def apply_proposal(proposal: Proposal, sch: te.Schedule) -> None:
    """Apply a Proposal to a Schedule, converting all the Plans into TE scheduling instructions.

    Note that the Schedule is mutated in-place.

    Parameters
    ----------
    proposal : Proposal
        The Proposal to apply to the Schedule.
    sch : te.Schedule
        The Schedule to apply to Proposal to.

    """
    for plan in proposal.plans:
        output_tensor_config = plan.output_config
        output_tensor = output_tensor_config.tensor
        output_part = output_tensor.producers[0]
        if output_part.in_line:
            continue
        stripe_config = output_tensor_config.stripe_configs[0]
        stripe_shape = [int(x) for x in stripe_config.shape]
        stripe_stage, stripe_axis = stripe_part(output_part, stripe_shape, sch)
        copy_te_tensors = []
        readers = defaultdict(list)
        for part in plan.part_group:
            if part != output_part:
                cascade_part(part, stripe_stage, stripe_axis, sch)

            update_readers(part, readers)
            for i, input_tensor in enumerate(part.input_tensors):
                tensor_config = plan.tensor_configs[input_tensor]
                if tensor_config.home_region != tensor_config.copy_region:
                    copy_te_tensors.append(part.subgraph.input_tensors[i])

        for te_tensor in copy_te_tensors:
            copy_stage = sch.cache_read(te_tensor, "global", readers[te_tensor])
            sch[copy_stage].compute_at(stripe_stage, stripe_axis)


def create_home_map(
    graph: CascaderGraph,
    io_region: MemoryRegion,
    constant_region: MemoryRegion,
    working_regions: List[MemoryRegion],
) -> Dict[Tensor, List[MemoryRegion]]:
    """Create a map between Tensors and the MemoryRegions they can be homed in."""
    home_map = {}
    for tensor in graph.tensor_order:
        if tensor.is_constant:
            home_map[tensor] = [constant_region]
        elif tensor in graph.input_tensors or tensor in graph.output_tensors:
            home_map[tensor] = [io_region]
        else:
            home_map[tensor] = working_regions

    return home_map


def choose_proposal(proposals: List[Proposal], cascade_region: MemoryRegion):
    """Choose the best performing Proposal that doesn't overflow the cascade region."""
    proposal_choice = proposals[0]
    for proposal in reversed(proposals):
        if proposal.memory_usage < cascade_region.size:
            proposal_choice = proposal
            break

    return proposal_choice


def cascade(
    sch: te.Schedule,
    te_graph: TESubgraph,
    const_dict: Dict[int, np.ndarray],
    options: CascaderOptions,
    io_region: MemoryRegion,
    constant_region: MemoryRegion,
    working_regions: List[MemoryRegion],
    device_config: EthosuDeviceConfig,
) -> None:
    """Schedule a Tensor Expression graph using the technique of 'cascading'.

    'Cascading' is a technique whereby operations are split into smaller
    dependent tiles ('stripes') which can then execute in an interleaved
    fashion. This allows for operations to execute together rather than
    sequentially which can reduce intermediate memory requirements and in
    certain cases improve performance.

    For more detail on 'cascading' as well as how it is implemented, refer to
    the RFC here: https://github.com/apache/tvm-rfcs/pull/37.

    Parameters
    ----------
    sch : te.Schedule
        The Schedule to apply the cascading to.
    te_graph : TESubgraph
        The Tensor Expression graph from which the Schedule was created.
    const_dict : Dict[int, np.ndarray]
        A dictionary mapping input index to constant data if that input is
        to be a constant.
    options : CascaderOptions
        Configuration options for the cascading scheduler.
    io_region : MemoryRegion
        The MemoryRegion in which input/output tensors should reside.
    constant_region : MemoryRegion
        The MemoryRegion in which constants should reside.
    working_regions : List[MemoryRegion]
        The MemoryRegions in which intermediate working tensors can reside. The
        cascading scheduler will select which MemoryRegion to per tensor.
    device_config : EthosuDeviceConfig
        Target device configuration.

    """
    assert options.cascade_region in working_regions
    # First convert the Tensor Expression graph into a CascaderGraph
    casc_graph = create_cascader_graph(te_graph, const_dict, device_config)
    # Then create a mapping between Tensors and their possible memory homes
    home_map = create_home_map(casc_graph, io_region, constant_region, working_regions)
    # Generate Proposals for Pareto-optimal ways to cascade the CascaderGraph
    proposals = generate_proposals(casc_graph, home_map, options)
    # Select the best Proposal subject to the memory constraints
    proposal_choice = choose_proposal(proposals, options.cascade_region)
    # Apply the selected Proposal to the Tensor Expression Schedule
    apply_proposal(proposal_choice, sch)
