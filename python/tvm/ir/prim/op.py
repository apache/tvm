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

from tvm.ir.base import UnknownLoc

from ..base import Location
from ..expr import Call, Expr
from . import _ffi_api
from .expr import ExprWithOp


def convert(expr) -> Expr:
    """Convert a scalar or sequence to primitive expressions."""
    return _ffi_api.convert(expr)


def min_value(dtype, loc=UnknownLoc()):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype, loc)  # type: ignore


def max_value(dtype: str, loc: Location = UnknownLoc()) -> Any:
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype, loc)  # type: ignore


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


def any(*args, loc=UnknownLoc()):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    loc : Location
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
    val = _ffi_api._OpOr(args[0], args[1], loc)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpOr(val, args[i], loc)  # type: ignore
    return val


def all(*args, loc=UnknownLoc()):
    """Create a new expression of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    loc : Location
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
    val = _ffi_api._OpAnd(args[0], args[1], loc)  # type: ignore
    for i in range(2, len(args)):
        val = _ffi_api._OpAnd(val, args[i], loc)  # type: ignore
    return val


def log2(x, *, ty=None, loc=UnknownLoc()):
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
    return Call("prim.log2", [x], ty=ty, loc=loc)


def ceil(x, loc=UnknownLoc()):
    """Take ceil of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.ceil(x, loc)  # type: ignore


def bitwise_and(x, y, loc=UnknownLoc()):
    """Take bitwise and of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_and(x, y, loc)


def bitwise_not(x, loc=UnknownLoc()):
    """Take bitwise not of input value

    Parameters
    ----------
    x : Expr
        Input operand

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_not(x, loc)


def bitwise_or(x, y, loc=UnknownLoc()):
    """Take bitwise or of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_or(x, y, loc)


def bitwise_xor(x, y, loc=UnknownLoc()):
    """Take bitwise xor of two values

    Parameters
    ----------
    x : Expr
        Left operand

    y : Expr
        Right operand

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    res : Expr
        The result.
    """
    return _ffi_api.bitwise_xor(x, y, loc)


def likely(cond, loc=UnknownLoc()):
    """Mark condition as likely.

    Parameters
    ----------

    cond : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The marked expression.
    """
    return _ffi_api.likely(cond, loc)  # type: ignore


def shift_left(x, y, loc=UnknownLoc()):
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
    return _ffi_api.left_shift(x, y, loc)


def shift_right(x, y, loc=UnknownLoc()):
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
    return _ffi_api.right_shift(x, y, loc)


def if_then_else(cond, t, f, loc=UnknownLoc()):
    """Conditional selection expression.

    Parameters
    ----------
    cond : Expr
        The condition

    t : Expr
        The result expression if cond is true.

    f : Expr
        The result expression if cond is false.

    loc : Location
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
    return _ffi_api._OpIfThenElse(cond, t, f, loc)  # type: ignore


def div(a, b, loc=UnknownLoc()):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    loc : Location
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b, loc).
    """
    return _ffi_api._OpDiv(a, b, loc)  # type: ignore


def indexdiv(a, b, loc=UnknownLoc()):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    loc : Location
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
    return _ffi_api._OpIndexDiv(a, b, loc)  # type: ignore


def indexmod(a, b, loc=UnknownLoc()):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : Expr
        The left hand operand, known to be non-negative.

    b : Expr
        The right hand operand, known to be non-negative.

    loc : Location
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
    return _ffi_api._OpIndexMod(a, b, loc)  # type: ignore


def truncdiv(a, b, loc=UnknownLoc()):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    loc : Location
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b, loc)  # type: ignore


def truncmod(a, b, loc=UnknownLoc()):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    loc : Location
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b, loc)  # type: ignore


def floordiv(a, b, loc=UnknownLoc()):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    loc : Location
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b, loc)  # type: ignore


def floormod(a, b, loc=UnknownLoc()):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : Expr
        The left hand operand

    b : Expr
        The right hand operand

    loc : Location
        The location of this operator in the source.

    Returns
    -------
    res : Expr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b, loc)  # type: ignore


def ceildiv(lhs, rhs, loc=UnknownLoc()):
    """Generic ceildiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.
    loc : Location
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of ceildiv operaton.
    """
    return _ffi_api._OpCeilDiv(lhs, rhs, loc)  # type: ignore


def vscale(*, ty=None, loc=UnknownLoc()):
    """Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
    (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
    Returns
    -------
    call : Expr
        Call to the vscale intrinsic
    """
    return Call("prim.vscale", [], ty=ty, loc=loc)


def min(a, b, loc=UnknownLoc()):
    """Elementwise minimum of two primitive expressions."""
    return _ffi_api._OpMin(a, b, loc)


def max(a, b, loc=UnknownLoc()):
    """Elementwise maximum of two primitive expressions."""
    return _ffi_api._OpMax(a, b, loc)


def _call_prim(ty, op, *args, loc=UnknownLoc()):
    return Call(op, args, ty=ty, loc=loc)


def _require_float_arg(op_name, x):
    x = convert(x)
    dtype = str(x.ty.dtype)
    if "float" not in dtype and "bfloat" not in dtype:
        raise TypeError(f"prim.{op_name} only supports floating-point inputs, but got {dtype}")
    return x


def assume(cond=None, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.assume", cond, loc=loc)


def infinity(dtype: str, loc: Location = UnknownLoc()) -> Any:
    """infinity value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.Expr
        The infinity value of dtype.
    """
    return _ffi_api.infinity(dtype, loc)  # type: ignore


def exp(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.exp", x, loc=loc)


def exp2(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.exp2", x, loc=loc)


def exp10(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.exp10", x, loc=loc)


def fma(x, y, z, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.fma", x, y, z, loc=loc)


def erf(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.erf", x, loc=loc)


def tanh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.tanh", x, loc=loc)


def sigmoid(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.sigmoid", x, loc=loc)


def log(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.log", x, loc=loc)


def log10(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.log10", x, loc=loc)


def log1p(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.log1p", x, loc=loc)


def tan(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.tan", x, loc=loc)


def cos(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.cos", x, loc=loc)


def cosh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.cosh", x, loc=loc)


def acos(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.acos", x, loc=loc)


def acosh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.acosh", x, loc=loc)


def sin(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.sin", x, loc=loc)


def sinh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.sinh", x, loc=loc)


def asin(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.asin", x, loc=loc)


def asinh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.asinh", x, loc=loc)


def atan(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.atan", x, loc=loc)


def atanh(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.atanh", x, loc=loc)


def atan2(x1, x2, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.atan2", x1, x2, loc=loc)


def sqrt(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.sqrt", x, loc=loc)


def rsqrt(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.rsqrt", x, loc=loc)


def floor(x: ExprWithOp, loc=UnknownLoc()):
    """Take floor of float input x.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.floor(x, loc)  # type: ignore


def trunc(x, loc=UnknownLoc()):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.trunc(x, loc)  # type: ignore


def abs(x, loc=UnknownLoc()):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.abs(x, loc)  # type: ignore


def round(x, loc=UnknownLoc()):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.round(x, loc)  # type: ignore


def nearbyint(x, loc=UnknownLoc()):
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

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.nearbyint(x, loc)  # type: ignore


def nextafter(x1, x2, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.nextafter", x1, x2, loc=loc)  # type: ignore


def hypot(x1, x2, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.hypot", x1, x2, loc=loc)  # type: ignore


def copysign(x1, x2, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.copysign", x1, x2, loc=loc)  # type: ignore


def ldexp(x1, x2, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.ldexp", x1, x2, loc=loc)  # type: ignore


def isnan(x, loc=UnknownLoc(), *, ty=None):
    """Check if input value is Nan.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return Call("prim.isnan", [x], ty=ty, loc=loc)


def isfinite(x, loc=UnknownLoc()):
    """Check if input value is finite.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isfinite(x, loc)  # type: ignore


def isinf(x, loc=UnknownLoc()):
    """Check if input value is infinite.

    Parameters
    ----------
    x : Expr
        Input argument.

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    y : Expr
        The result.
    """
    return _ffi_api.isinf(x, loc)  # type: ignore


def power(x, y, loc=UnknownLoc()):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, loc)  # type: ignore


def popcount(x, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.popcount", x, loc=loc)


def fmod(x, y, *, ty=None, loc=UnknownLoc()):
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
    return _call_prim(ty, "prim.fmod", x, y, loc=loc)


def pow(x, y, loc=UnknownLoc()):
    """x power y

    Parameters
    ----------
    x : Expr
        Input argument.

    y : Expr
        The exponent

    loc : Location
        The location of this operator in the source code.

    Returns
    -------
    z : Expr
        The result.
    """
    return _ffi_api._OpPow(x, y, loc)  # type: ignore
