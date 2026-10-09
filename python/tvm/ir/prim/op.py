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
"""Construction helpers for shared primitive expressions."""

from typing import Any

from ..base import Span
from ..expr import Call, Expr
from . import _ffi_api
from .expr import ExprWithOp


def convert(expr) -> Expr:
    """Convert a scalar or sequence to primitive expressions."""
    return _ffi_api.convert(expr)


def min_value(dtype, span=None):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype, span)  # type: ignore


def max_value(dtype: str, span: Span | None = None) -> Any:
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype, span)  # type: ignore


def clz(x):
    """Count leading zero bits of an integer x.

    Parameters
    ----------
    x : Expr
        Input 32 or 64 bit integer.
        The result is undefined if the input is 0.

    Returns
    -------
    y : Expr
        The result.
    """
    return Call("prim.clz", [x], ty="int32")


def any(*args, span=None):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    val = _ffi_api._OpOr(args[0], args[1], span)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpOr(val, args[i], span)  # type: ignore
    return val


def all(*args, span=None):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    val = _ffi_api._OpAnd(args[0], args[1], span)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpAnd(val, args[i], span)  # type: ignore
    return val


def log2(x, *, ty=None, span=None):
    """Take log2 of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return Call("prim.log2", [x], ty=ty, span=span)


def ceil(x, span=None):
    """Take ceil of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.ceil(x, span)  # type: ignore


def bitwise_and(x, y, span=None):
    """Take bitwise and of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_and(x, y, span)


def bitwise_not(x, span=None):
    """Take bitwise not of input value

    Parameters
    ----------
    x : Expr
        Input operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_not(x, span)


def bitwise_or(x, y, span=None):
    """Take bitwise or of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_or(x, y, span)


def bitwise_xor(x, y, span=None):
    """Take bitwise xor of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_xor(x, y, span)


def likely(cond, span=None):
    """Mark condition as likely.

    Parameters
    ----------

    cond : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The marked expression.
    """
    return _ffi_api.likely(cond, span)  # type: ignore


def shift_left(x, y, span=None):
    """Return the result of x left shifted by y bits.

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        Input argument.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api.left_shift(x, y, span)


def shift_right(x, y, span=None):
    """Return the result of x right shifted by y bits.

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        Input argument.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api.right_shift(x, y, span)


def if_then_else(cond, t, f, span=None):
    """Conditional selection expression.

    Parameters
    ----------
    cond : Expr
        The condition

    t : Expr
        The result expression if cond is true.

    f : Expr
        The result expression if cond is false.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    result : Node
        The result of conditional expression.

    Note
    ----
    Unlike Select, if_then_else will not execute
    the branch that does not satisfy the condition.
    You can use it to guard against out of bound access.
    Unlike Select, if_then_else cannot be vectorized
    if some lanes in the vector have different conditions.
    """
    return _ffi_api._OpIfThenElse(cond, t, f, span)  # type: ignore


def div(a, b, span=None):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b, span).
    """
    return _ffi_api._OpDiv(a, b, span)  # type: ignore


def indexdiv(a, b, span=None):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexDiv(a, b, span)  # type: ignore


def indexmod(a, b, span=None):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexMod(a, b, span)  # type: ignore


def truncdiv(a, b, span=None):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b, span)  # type: ignore


def truncmod(a, b, span=None):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b, span)  # type: ignore


def floordiv(a, b, span=None):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b, span)  # type: ignore


def floormod(a, b, span=None):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b, span)  # type: ignore


def ceildiv(lhs, rhs, span=None):
    """Generic ceildiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of ceildiv operaton.
    """
    return _ffi_api._OpCeilDiv(lhs, rhs, span)  # type: ignore


def vscale(*, ty=None, span=None):
    """Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
    (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
    Returns
    -------
    call : Expr
        Call to the vscale intrinsic
    """
    return Call("prim.vscale", [], ty=ty, span=span)


def min(a, b, span=None):
    """Elementwise minimum of two primitive expressions."""
    return _ffi_api._OpMin(a, b, span)


def max(a, b, span=None):
    """Elementwise maximum of two primitive expressions."""
    return _ffi_api._OpMax(a, b, span)


def _call_prim(ty, op, *args, span=None):
    return Call(op, args, ty=ty, span=span)


def _require_float_arg(op_name, x):
    x = convert(x)
    dtype = str(x.ty.dtype)
    if "float" not in dtype and "bfloat" not in dtype:
        raise TypeError(f"prim.{op_name} only supports floating-point inputs, but got {dtype}")
    return x


def assume(cond=None, *, ty=None, span=None):
    """Provide a true statement that can be used for simplifications

    Parameters
    ----------
    cond : Expr
       The constraint condition.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return _call_prim(ty, "prim.assume", cond, span=span)


def infinity(dtype: str, span: Span | None = None) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The infinity value of dtype.
    """
    return _ffi_api.infinity(dtype, span)  # type: ignore


def exp(x, *, ty=None, span=None):
    """Take exponential of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.exp", x, span=span)


def exp2(x, *, ty=None, span=None):
    """Calculate 2**x

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.exp2", x, span=span)


def exp10(x, *, ty=None, span=None):
    """Calculate 10**x

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.exp10", x, span=span)


def fma(x, y, z, *, ty=None, span=None):
    """Take fused multiply-add of input x, y, z.

    Parameters
    ----------
    x : Expr
        First input argument.

    y : Expr
        Second input argument.

    z : Expr
        Third input argument.

    Returns
    -------
    out : Expr
        The result of x * y + z.
    """
    x = convert(x)
    y = convert(y)
    z = convert(z)
    return _call_prim(ty, "prim.fma", x, y, z, span=span)


def erf(x, *, ty=None, span=None):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.erf", x, span=span)


def tanh(x, *, ty=None, span=None):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.tanh", x, span=span)


def sigmoid(x, *, ty=None, span=None):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.sigmoid", x, span=span)


def log(x, *, ty=None, span=None):
    """Take log of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.log", x, span=span)


def log10(x, *, ty=None, span=None):
    """Take log10 of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.log10", x, span=span)


def log1p(x, *, ty=None, span=None):
    """Take log(x + 1) with respect to input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.log1p", x, span=span)


def tan(x, *, ty=None, span=None):
    """Take tan of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("tan", x)
    return _call_prim(ty, "prim.tan", x, span=span)


def cos(x, *, ty=None, span=None):
    """Take cos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("cos", x)
    return _call_prim(ty, "prim.cos", x, span=span)


def cosh(x, *, ty=None, span=None):
    """Take cosh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.cosh", x, span=span)


def acos(x, *, ty=None, span=None):
    """Take acos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.acos", x, span=span)


def acosh(x, *, ty=None, span=None):
    """Take acos of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.acosh", x, span=span)


def sin(x, *, ty=None, span=None):
    """Take sin of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = _require_float_arg("sin", x)
    return _call_prim(ty, "prim.sin", x, span=span)


def sinh(x, *, ty=None, span=None):
    """Take sinh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.sinh", x, span=span)


def asin(x, *, ty=None, span=None):
    """Take asin of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.asin", x, span=span)


def asinh(x, *, ty=None, span=None):
    """Take asinh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.asinh", x, span=span)


def atan(x, *, ty=None, span=None):
    """Take atan of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.atan", x, span=span)


def atanh(x, *, ty=None, span=None):
    """Take atanh of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.atanh", x, span=span)


def atan2(x1, x2, *, ty=None, span=None):
    """Take arctan2(x1, x2).

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = convert(x1)
    x2 = convert(x2)
    return _call_prim(ty, "prim.atan2", x1, x2, span=span)


def sqrt(x, *, ty=None, span=None):
    """Take square root of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.sqrt", x, span=span)


def rsqrt(x, *, ty=None, span=None):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.rsqrt", x, span=span)


def floor(x: ExprWithOp, span=None):
    """Take floor of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.floor(x, span)  # type: ignore


def trunc(x, span=None):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.trunc(x, span)  # type: ignore


def abs(x, span=None):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.abs(x, span)  # type: ignore


def round(x, span=None):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.round(x, span)  # type: ignore


def nearbyint(x, span=None):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from te.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas te.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.nearbyint(x, span)  # type: ignore


def nextafter(x1, x2, *, ty=None, span=None):
    """Return the next floating-point value after x1 towards x2.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = convert(x1)
    x2 = convert(x2)
    return _call_prim(ty, "prim.nextafter", x1, x2, span=span)  # type: ignore


def hypot(x1, x2, *, ty=None, span=None):
    """Equivalent to sqrt(x1**2 + x2**2), element-wise.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = convert(x1)
    x2 = convert(x2)
    return _call_prim(ty, "prim.hypot", x1, x2, span=span)  # type: ignore


def copysign(x1, x2, *, ty=None, span=None):
    """Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = convert(x1)
    x2 = convert(x2)
    return _call_prim(ty, "prim.copysign", x1, x2, span=span)  # type: ignore


def ldexp(x1, x2, *, ty=None, span=None):
    """Returns x1 * (2 ** x2).

    Parameters
    ----------
    x1 : Expr
        Input argument.

    x2 : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x1 = convert(x1)
    x2 = convert(x2)
    return _call_prim(ty, "prim.ldexp", x1, x2, span=span)  # type: ignore


def isnan(x, span=None):
    """Check if input value is Nan.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isnan(x, span)  # type: ignore


def isfinite(x, span=None):
    """Check if input value is finite.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isfinite(x, span)  # type: ignore


def isinf(x, span=None):
    """Check if input value is infinite.

    Parameters
    ----------
    x : Expr
        Input argument.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isinf(x, span)  # type: ignore


def power(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore


def popcount(x, *, ty=None, span=None):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    Returns
    -------
    y : Expr
        The result.
    """
    x = convert(x)
    return _call_prim(ty, "prim.popcount", x, span=span)


def fmod(x, y, *, ty=None, span=None):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : Expr
        Input argument.
    y : Expr
        Input argument.

    Returns
    -------
    z : Expr
        The result.
    """
    x = convert(x)
    y = convert(y)
    return _call_prim(ty, "prim.fmod", x, y, span=span)


def pow(x, y, span=None):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, span)  # type: ignore
