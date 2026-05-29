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

"""Shared helpers for copy_async operator dispatch variants.

The TMA-specific lowering moved to ``tma.py``. What remains here are the tiny
layout helpers other variants (e.g. ``dsmem.py``) still import.
"""

from tvm.arith import Analyzer
from tvm.tirx.layout import ComposeLayout, Layout, S, SwizzleLayout, TileLayout


def find_contiguous_region(layout: TileLayout) -> tuple:
    """Return the maximal stride-1 contiguous memory-shard chain.

    Starts from stride==1 and repeatedly picks the shard whose stride equals
    the running product of extents, stopping when no shard matches. Returns
    the maximal chain; callers that need a shorter prefix should take one
    themselves (e.g. to satisfy TMA's rank<=5 or a per-path reduction step).
    Stride/extent comparisons go through an ``Analyzer`` so symbolic strides
    work.
    """

    analyzer = Analyzer()
    memory_shards = [
        (i, s)
        for i, s in enumerate(layout.shard)
        if s.axis.is_memory() and not analyzer.can_prove_equal(s.extent, 1)
    ]
    if not memory_shards:
        return [], 1

    contiguous_indices: list[int] = []
    contiguous_extent = 1
    expected_stride = 1
    consumed: set[int] = set()

    while True:
        for idx, shard in memory_shards:
            if idx in consumed:
                continue
            if analyzer.can_prove_equal(shard.stride, expected_stride):
                consumed.add(idx)
                contiguous_indices.append(idx)
                contiguous_extent *= shard.extent
                expected_stride = contiguous_extent
                break
        else:
            break

    if not contiguous_indices:
        return [], 0
    return contiguous_indices, contiguous_extent


def to_tile_layout(layout: Layout, shape: list[int]) -> TileLayout:
    """Normalize any layout kind to a TileLayout for pointer arithmetic."""

    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    if isinstance(layout, SwizzleLayout):
        return TileLayout(S[tuple(shape)])
    return layout
