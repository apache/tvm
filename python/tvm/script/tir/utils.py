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
"""Helper functions in TVM Script Parser"""

from typing import List, Optional, Union

from tvm.arith import Analyzer
from tvm.ir import Range
from tvm.tir import PrimExpr, BufferRegion
from .node import BufferSlice


def buffer_slice_to_region(
    buffer_slice: BufferSlice, analyzer: Optional[Analyzer] = None
) -> BufferRegion:
    """Construct BufferRegion from BufferSlice

    Parameters
    ----------
    buffer_slice : BufferSlice
        The input BufferSlice

    analyzer : Optional[tvm.arith.Analyzer]
        The analyzer for simplifying. If not provided, the method will construct a new one

    Returns
    -------
    buffer_region : BufferRegion
        The constructed BufferRegion.
    """
    region: List[Range] = []
    for s in buffer_slice.slices:
        start: Union[PrimExpr, int] = s.start
        extent: Union[PrimExpr, int] = 1 if s.stop is None else s.stop - s.start
        if not analyzer:
            analyzer = Analyzer()
        if isinstance(extent, PrimExpr):
            extent = analyzer.simplify(extent)
        region.append(Range.from_min_extent(start, extent, span=s.span))
    return BufferRegion(buffer_slice.buffer, region)
