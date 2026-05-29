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

"""Dimension mapping utilities for TRN operator scheduling."""

from collections import namedtuple

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion

# Represents the part of data iter covered by the buffer region
RangeInfo = namedtuple(
    "RangeInfo", ["start", "extent", "dim_in_data_iter", "dim_in_shape", "dim_type"]
)


def normalize_and_group(layout, shape):
    """Normalize a layout with a given shape.

    Parameters
    ----------
    layout : Union[Tx.TrainiumLayout, Tx.TileLayout]
        The layout to normalize
    shape : List[int]
        The shape to normalize with

    Returns
    -------
    Tuple[Union[Tx.TrainiumLayout, Tx.TileLayout], List[int]] :
        Normalized layout and separators

    Raises
    ------
    ValueError :
        If layout is not a valid layout type
    """
    if isinstance(layout, Tx.TileLayout):
        return layout.canonicalize().group(shape)
    else:
        raise ValueError("Invalid layout")


def get_ewise_dim_map(
    buffer_region: BufferRegion, second_buffer_region: BufferRegion, analyzer: Analyzer
):
    """Get the dimension map between two elementwise buffer regions.

    Parameters
    ----------
    buffer_region : BufferRegion
        The first buffer region
    second_buffer_region : BufferRegion
        The second buffer region
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Dict[int, int] :
        A dimension map from first to second buffer region

    Raises
    ------
    AssertionError :
        If dimensions do not match
    """
    extent_1 = [r.extent for r in buffer_region.region]
    extent_2 = [r.extent for r in second_buffer_region.region]
    extent_1_non_unit = [e for e in extent_1 if e != 1]
    extent_2_non_unit = [e for e in extent_2 if e != 1]
    assert all(
        [
            len(extent_1_non_unit) == len(extent_2_non_unit),
            all(
                analyzer.can_prove_equal(s, d) for s, d in zip(extent_1_non_unit, extent_2_non_unit)
            ),
        ]
    )
    dim_map = {}
    i = 0
    j = 0
    while i < len(extent_1) and j < len(extent_2):
        if analyzer.can_prove_equal(extent_1[i], 1):
            i += 1
            continue
        if analyzer.can_prove_equal(extent_2[j], 1):
            j += 1
            continue
        dim_map[i] = j
        i += 1
        j += 1
    return dim_map


def get_reduction_dim_map(
    src_buffer_region: BufferRegion,
    dst_buffer_region: BufferRegion,
    axes: tuple[int],
    analyzer: Analyzer,
):
    """Get the dimension map between source and destination buffer regions for reduction.

    Parameters
    ----------
    src_buffer_region : BufferRegion
        The source buffer region
    dst_buffer_region : BufferRegion
        The destination buffer region
    axes : Tuple[int]
        The reduction axes
    analyzer : Analyzer
        The analyzer to use

    Returns
    -------
    Dict[int, int] :
        A dimension map from source to destination buffer region

    Raises
    ------
    AssertionError :
        If dimensions do not match
    """
    dst_region = dst_buffer_region.region
    dst_extent = [r.extent for r in dst_region]
    dst_non_unit_extent_ = [(i, e) for i, e in enumerate(dst_extent) if e != 1]
    src_region = src_buffer_region.region
    src_extent = [r.extent for r in src_region]
    src_non_unit_extent_ = [(i, e) for i, e in enumerate(src_extent) if e != 1]
    src_non_reduction_extents = [(i, e) for i, e in src_non_unit_extent_ if i not in axes]
    assert len(src_non_reduction_extents) == len(dst_non_unit_extent_), (
        f"Source and destination must have the same number of non-reduction extents: {len(src_non_reduction_extents)} != {len(dst_non_unit_extent_)}"  # noqa: E501
    )
    for i in range(len(src_non_reduction_extents)):
        assert analyzer.can_prove_equal(
            src_non_reduction_extents[i][1], dst_non_unit_extent_[i][1]
        ), (
            f"Source and destination must have the same extent for non-reduction axes: {src_non_reduction_extents[i][1]} != {dst_non_unit_extent_[i][1]}"  # noqa: E501
        )
    dim_map = {s[0]: d[0] for s, d in zip(src_non_reduction_extents, dst_non_unit_extent_)}
    return dim_map


class DimensionMapper:
    """
    A class to manage dimension mappings between tensors.

    A dimension mapping (dim_map) has type Dict[int, int]. dim_map[i] = j means
    dimension i in the first tensor should be mapped to dimension j in the second tensor.
    """

    def __init__(self):
        self.mappings = {}  # Dictionary to store mappings between tensors

    def register_dim_map(self, first_tensor, second_tensor, dim_map):
        """
        Register a dimension mapping between two tensors.

        Args:
            first_tensor: The first tensor
            second_tensor: The second tensor
            dim_map: A dictionary mapping dimensions from first_tensor to second_tensor
        """
        # Initialize dictionaries if they don't exist
        if first_tensor not in self.mappings:
            self.mappings[first_tensor] = {}

        # Register the mapping
        self.mappings[first_tensor][second_tensor] = dim_map

        # Register the reverse mapping
        reverse_dim_map = {dim_map[i]: i for i in dim_map}

        if second_tensor not in self.mappings:
            self.mappings[second_tensor] = {}

        self.mappings[second_tensor][first_tensor] = reverse_dim_map

    def compose_mappings(self, map1, map2):
        """
        Compose two mappings: map1 followed by map2.

        Args:
            map1: The first mapping
            map2: The second mapping

        Returns:
            A composition of the two mappings, or None if the composition is empty
        """
        result = {}
        for i, j in map1.items():
            if j in map2:
                result[i] = map2[j]

        # If the result is empty, return None
        return result if result else None

    def get_dim_map(self, first_tensor, second_tensor):
        """
        Get the dimension mapping between two tensors.

        Args:
            first_tensor: The first tensor
            second_tensor: The second tensor

        Returns:
            A dictionary mapping dimensions from first_tensor to second_tensor,
            or {} if no mapping exists
        """
        # Check if there is a direct mapping
        if first_tensor in self.mappings and second_tensor in self.mappings[first_tensor]:
            return self.mappings[first_tensor][second_tensor]

        # No direct mapping, try to find a path using BFS
        visited = {first_tensor}
        queue = []

        # Add all direct neighbors of the first tensor to the queue
        if first_tensor in self.mappings:
            for neighbor, direct_mapping in self.mappings[first_tensor].items():
                visited.add(neighbor)
                queue.append((neighbor, direct_mapping))

        while queue:
            current_tensor, mapping_from_first = queue.pop(0)

            if current_tensor == second_tensor:
                # Found a path to the second tensor
                self.register_dim_map(first_tensor, second_tensor, mapping_from_first)
                return mapping_from_first

            if current_tensor not in self.mappings:
                continue

            for neighbor, direct_mapping in self.mappings[current_tensor].items():
                if neighbor not in visited:
                    visited.add(neighbor)

                    # Compose the mappings: first_tensor -> current_tensor -> neighbor
                    composed_mapping = self.compose_mappings(mapping_from_first, direct_mapping)

                    # Only add to the queue if the composed mapping is not None
                    if composed_mapping is not None:
                        queue.append((neighbor, composed_mapping))

        # No mapping found
        return {}
