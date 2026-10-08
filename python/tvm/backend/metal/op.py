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

from tvm.ir import Call, Expr, Op, Var
from tvm.ir.op import _make_op_api
from tvm.tirx import is_tensor_var
from tvm.tirx.op import call_intrin


def make_filled_simdgroup_matrix(d, index, value, col=8, row=8, *, ty=None, span=None):
    """Create a filled SIMDGroup matrix."""

    return Call(
        "tirx.metal.make_filled_simdgroup_matrix",
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
        "tirx.metal.simdgroup_load",
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
        "tirx.metal.simdgroup_store",
        [d, index, ptr, stride, col, row, transpose_matrix],
        ty=ty,
        span=span,
    )


def simdgroup_multiply_accumulate(
    d, index_d, a, index_a, b, index_b, c, index_c, *, ty=None, span=None
):
    """Multiply and accumulate two matrices in simdgroup."""

    return Call(
        "tirx.metal.simdgroup_multiply_accumulate",
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
    "cooperative_tensor_fill",
    "cooperative_tensor_load",
    "cooperative_tensor_multiply_accumulate",
    "cooperative_tensor_store",
    "make_filled_simdgroup_matrix",
    "simd_shuffle",
    "simd_shuffle_down",
    "simd_shuffle_up",
    "simdgroup_load",
    "simdgroup_multiply_accumulate",
    "simdgroup_store",
]


def cooperative_tensor_fill(
    d: Var,
    index: Expr,
    value: Expr,
    rows: int,
    cols: int,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.metal.cooperative_tensor_fill",
        d,
        index,
        value,
        rows,
        cols,
        span=span,
    )


def cooperative_tensor_load(
    d: Var,
    index: Expr,
    ptr: Expr,
    stride: Expr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.metal.cooperative_tensor_load",
        d,
        index,
        ptr,
        stride,
        rows,
        cols,
        transpose_matrix,
        mma_M,
        mma_N,
        mma_K,
        operand_role,
        span=span,
    )


def cooperative_tensor_store(
    d: Expr,
    index: Expr,
    ptr: Expr,
    stride: Expr,
    rows: int,
    cols: int,
    transpose_matrix: bool = False,
    mma_M: int = 0,
    mma_N: int = 0,
    mma_K: int = 0,
    operand_role: int = 0,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.metal.cooperative_tensor_store",
        d,
        index,
        ptr,
        stride,
        rows,
        cols,
        transpose_matrix,
        mma_M,
        mma_N,
        mma_K,
        operand_role,
        span=span,
    )


def cooperative_tensor_multiply_accumulate(
    d: Var,
    index_d: Expr,
    a: Var,
    index_a: Expr,
    b: Var,
    index_b: Expr,
    c: Var,
    index_c: Expr,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    *,
    ty=None,
    span=None,
):
    return call_intrin(
        ty,
        "tirx.metal.cooperative_tensor_multiply_accumulate",
        d,
        index_d,
        a,
        index_a,
        b,
        index_b,
        c,
        index_c,
        M,
        N,
        K,
        transpose_a,
        transpose_b,
        span=span,
    )
