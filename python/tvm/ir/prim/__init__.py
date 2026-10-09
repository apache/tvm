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
# pylint: disable=unused-import
"""Primitive expressions and construction helpers shared by TVM IR dialects."""

from .._constant import const
from ..expr import Expr
from . import _ffi_api
from .expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    BinaryOpExpr,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    Broadcast,
    Cast,
    CmpExpr,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    Let,
    LogicalExpr,
    LShift,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    Ramp,
    RShift,
    Select,
    Shuffle,
    Sub,
)
from .op import (
    abs,
    acos,
    acosh,
    all,
    any,
    asin,
    asinh,
    assume,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    ceil,
    ceildiv,
    clz,
    convert,
    copysign,
    cos,
    cosh,
    div,
    erf,
    exp,
    exp2,
    exp10,
    floor,
    floordiv,
    floormod,
    fma,
    fmod,
    hypot,
    if_then_else,
    indexdiv,
    indexmod,
    infinity,
    isfinite,
    isinf,
    isnan,
    ldexp,
    likely,
    log,
    log1p,
    log2,
    log10,
    max,
    max_value,
    min,
    min_value,
    nearbyint,
    nextafter,
    popcount,
    pow,
    power,
    round,
    rsqrt,
    shift_left,
    shift_right,
    sigmoid,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    truncdiv,
    truncmod,
    vscale,
)


def expr_deep_equal(lhs: Expr, rhs: Expr) -> bool:
    """Deeply compare two nested expressions.

    Parameters
    ----------
    lhs : Expr
        The left operand.

    rhs : Expr
        The right operand.

    Returns
    -------
    result : bool
        The comparison result

    Note
    ----

    This function does not remap variable bindings, it will not
    return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
    Use py:func:`tvm_ffi.structural_equal` to handle structural variable remapping.

    Due to the restriction of not remapping variables, this function can run
    faster than StructuralEqual and can be used as a utility function during arithmetic
    simplifications.

    Always consider py:func:`tvm_ffi.structural_equal` first, which handles
    the structural remapping.

    See Also
    --------
    tvm_ffi.structural_equal
    """
    return _ffi_api.expr_deep_equal(lhs, rhs)  # type: ignore
