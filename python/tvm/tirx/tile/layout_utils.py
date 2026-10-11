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
"""Layouts inferred only while lowering a tile instruction."""

from tvm.sym import Analyzer
from tvm.tirx.layout import S, TileLayout


def effective_layout(tensor):
    """Return an explicit layout or the compact layout of a global tensor.

    Inference uses the complete source shape before a region is sliced. The
    source type and its pointer, strides and element offset remain unchanged.
    Other storage scopes and noncompact global tensors require an explicit layout.
    """
    if tensor.layout is not None:
        return tensor.layout
    if tensor.scope() != "global":
        raise ValueError(
            f"tile instruction requires an explicit layout for {tensor.scope()} tensor"
        )
    analyzer = Analyzer()
    expected = 1
    if tensor.strides:
        for extent, stride in reversed(list(zip(tensor.shape, tensor.strides))):
            if not analyzer.can_prove_equal(extent, 1) and not analyzer.can_prove_equal(
                stride, expected
            ):
                raise ValueError(
                    "tile instruction requires an explicit layout for a noncompact global tensor"
                )
            expected = analyzer.simplify(expected * extent)
    return TileLayout(S[tuple(tensor.shape)])
