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
"""Trainium-specific TIRx layout helpers."""

from __future__ import annotations

import functools
import operator
import re

import tvm
from tvm.tirx.expr import PrimExpr
from tvm.tirx.layout import Axis, Iter, Layout, S, TileLayout

_TRN_MEMORY_AXES = {"F", "P", "Bank"}
_PSUM_MAX_ELEM_PER_BANK = 512


def is_trainium_layout(layout: Layout | None) -> bool:
    """Return whether a layout uses only Trainium memory axes."""
    if not isinstance(layout, TileLayout):
        return False
    return not any(
        iter.axis.is_memory() and iter.axis.name not in _TRN_MEMORY_AXES for iter in layout.shard
    )


def trainium_layout(annotation: str, shape: tuple[PrimExpr], is_psum: bool = False) -> TileLayout:
    """Create a Trainium tile layout from a PF annotation string and logical shape."""
    analyzer = tvm.arith.Analyzer()
    assert re.fullmatch(r"[PF]*", annotation), (
        f"annotation {annotation} must be a string of 'P' and 'F'"
    )
    assert len(annotation) == len(shape), (
        f"annotation {annotation} and shape {shape} must have the same length"
    )
    num_p_dim = annotation.count("P")
    if num_p_dim == 1:
        p_idx = annotation.index("P")
        p_dim = shape[p_idx]
        assert analyzer.can_prove(p_dim <= 128 or p_dim % 128 == 0), (
            f"There is only 1 P in the annotation. Partition size {p_dim} must be less than "
            "or equal to 128 or a multiple of 128"
        )
        if analyzer.can_prove(p_dim > 128):
            annotation = "F" + annotation
            shape = (p_dim // 128, *shape[:p_idx], 128, *shape[p_idx + 1 :])
    elif num_p_dim > 1:
        p_dim_prod = functools.reduce(
            operator.mul, [s for s, c in zip(shape, annotation) if c == "P"]
        )
        assert analyzer.can_prove(p_dim_prod <= 128), (
            f"There are {num_p_dim} Ps in the annotation. Partition size {p_dim_prod} must be "
            "less than or equal to 128"
        )

    f_shape = [s for s, c in zip(shape, annotation) if c == "F"]
    p_shape = [s for s, c in zip(shape, annotation) if c == "P"]
    f_strides = Layout._get_default_strides(f_shape, 1)  # pylint: disable=protected-access
    p_strides = Layout._get_default_strides(p_shape, 1)  # pylint: disable=protected-access
    f_tile_layout = TileLayout(S[tuple(f_shape) : tuple(s @ Axis.F for s in f_strides)])
    p_tile_layout = TileLayout(S[tuple(p_shape) : tuple(s @ Axis.P for s in p_strides)])
    result = []
    f_index = p_index = 0

    for char in annotation:
        if char == "F":
            result.append(f_tile_layout.shard[f_index])
            f_index += 1
        else:
            result.append(p_tile_layout.shard[p_index])
            p_index += 1
    if num_p_dim == 1 and analyzer.can_prove(p_dim > 128):
        higher_p = result[0]
        result = result[1:]
        result = [*result[:p_idx], higher_p, *result[p_idx:]]

    res = TileLayout.from_iters(result, [], {})
    if is_psum:
        res = to_psum_layout(res)
    return res


def to_psum_layout(layout: TileLayout) -> TileLayout:
    """Convert a Trainium sbuf layout to its psum physical-bank layout."""
    analyzer = tvm.arith.Analyzer()
    shard = []
    for iter in layout.shard:
        if iter.axis.name == "F":
            if analyzer.can_prove(iter.stride % _PSUM_MAX_ELEM_PER_BANK == 0):
                stride = analyzer.simplify(iter.stride // _PSUM_MAX_ELEM_PER_BANK)
                shard.append(Iter(iter.extent, stride, Axis.get("Bank")))
            elif analyzer.can_prove(_PSUM_MAX_ELEM_PER_BANK % iter.stride == 0):
                c = analyzer.simplify(_PSUM_MAX_ELEM_PER_BANK // iter.stride)
                if analyzer.can_prove(iter.extent < c):
                    shard.append(iter)
                elif analyzer.can_prove(iter.extent % c == 0):
                    shard.append(Iter(analyzer.simplify(iter.extent // c), 1, Axis.get("Bank")))
                    shard.append(Iter(c, iter.stride, Axis.get("F")))
                else:
                    raise ValueError(f"layout {layout} can not be converted to psum layout")
            else:
                raise ValueError(f"layout {layout} can not be converted to psum layout")
        else:
            shard.append(iter)
    return TileLayout.from_iters(shard, [], {})


__all__ = ["is_trainium_layout", "to_psum_layout", "trainium_layout"]
