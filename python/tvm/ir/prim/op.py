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


def log2(x):
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
    return Call("prim.log2", [x], ty=x.ty)


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


def vscale():
    """Get the target's vscale value. It will be lowered to llvm.vscale intrinsic
    (https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)
    Returns
    -------
    call : Expr
        Call to the vscale intrinsic
    """
    return Call("prim.vscale", [], ty="int32")


def min(a, b, span=None):
    """Elementwise minimum of two primitive expressions."""
    return _ffi_api._OpMin(a, b, span)


def max(a, b, span=None):
    """Elementwise maximum of two primitive expressions."""
    return _ffi_api._OpMax(a, b, span)
