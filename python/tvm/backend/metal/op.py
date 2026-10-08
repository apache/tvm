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
"""Metal-owned TIR intrinsic builders."""

from __future__ import annotations

from tvm.ir import Call, Op
from tvm.ir.op import _make_op_api
from tvm.tirx import is_tensor_var


def make_filled_simdgroup_matrix(d, index, value, col=8, row=8, *, ty=None, span=None):
    """Create a filled SIMDGroup matrix."""

    return Call(
        "tirx.make_filled_simdgroup_matrix",
        [d, index, value, col, row],
        ty=ty,
        span=span,
    )


def simdgroup_load(
    d,
    index,
    ptr,
    stride,
    col=8,
    row=8,
    transpose_matrix=False,
    *,
    ty=None,
    span=None,
):
    """Load data from device or threadgroup memory to simdgroup."""

    return Call(
        "tirx.simdgroup_load",
        [d, index, ptr, stride, col, row, transpose_matrix],
        ty=ty,
        span=span,
    )


def simdgroup_store(
    d,
    index,
    ptr,
    stride,
    col=8,
    row=8,
    transpose_matrix=False,
    *,
    ty=None,
    span=None,
):
    """Store data from simdgroup to device or threadgroup memory."""

    return Call(
        "tirx.simdgroup_store",
        [d, index, ptr, stride, col, row, transpose_matrix],
        ty=ty,
        span=span,
    )


def simdgroup_multiply_accumulate(
    d, index_d, a, index_a, b, index_b, c, index_c, *, ty=None, span=None
):
    """Multiply and accumulate two matrices in simdgroup."""

    return Call(
        "tirx.simdgroup_multiply_accumulate",
        [d, index_d, a, index_a, b, index_b, c, index_c],
        ty=ty,
        span=span,
    )


_simd_shuffle = _make_op_api(Op.get("tirx.metal.simd_shuffle"), __name__)
_simd_shuffle_up = _make_op_api(Op.get("tirx.metal.simd_shuffle_up"), __name__)
_simd_shuffle_down = _make_op_api(Op.get("tirx.metal.simd_shuffle_down"), __name__)


def simd_shuffle(var, lane, *, ty=None, span=None):
    """Shuffle a value from the selected SIMD lane."""
    if is_tensor_var(var):
        var = var[0]
    return _simd_shuffle(var, lane, ty=ty, span=span)


def simd_shuffle_up(var, delta, *, ty=None, span=None):
    """Shuffle a value from a lower SIMD lane."""
    if is_tensor_var(var):
        var = var[0]
    return _simd_shuffle_up(var, delta, ty=ty, span=span)


def simd_shuffle_down(var, delta, *, ty=None, span=None):
    """Shuffle a value from a higher SIMD lane."""
    if is_tensor_var(var):
        var = var[0]
    return _simd_shuffle_down(var, delta, ty=ty, span=span)


__all__ = [
    "make_filled_simdgroup_matrix",
    "simd_shuffle",
    "simd_shuffle_down",
    "simd_shuffle_up",
    "simdgroup_load",
    "simdgroup_multiply_accumulate",
    "simdgroup_store",
]
