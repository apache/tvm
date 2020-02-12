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
# pylint: disable=redefined-builtin
"""Operators used in TIR expression."""
import tvm._ffi
from tvm.runtime import convert, const
from tvm.schedule import Buffer

from .expr import Call
from . import _ffi_api


def _pack_buffer(buf):
    """Build intrinsics that packs the buffer.
    """
    assert buf.shape
    shape = Call("handle", "tvm_stack_make_shape", buf.shape,
                 Call.Intrinsic, None, 0)
    strides = Call("handle", "tvm_stack_make_shape", buf.strides,
                   Call.Intrinsic, None, 0) if buf.strides else 0
    pack_args = [buf.data,
                 shape,
                 strides,
                 len(buf.shape),
                 const(0, dtype=buf.dtype),
                 buf.elem_offset]
    return Call("handle", "tvm_stack_make_array",
                pack_args, Call.Intrinsic, None, 0)

def call_packed(*args):
    """Build expression by call an external packed function.

    The argument to packed function can be Expr or Buffer.
    The argument is the corresponding POD type when Expr is presented.

    When the argument is Buffer, the corresponding PackedFunc
    will recieve an TVMArrayHandle whose content is valid during the callback period.
    If the PackedFunc is a python callback, then the corresponding argument is NDArray.

    Parameters
    ----------
    args : list of Expr or Buffer.
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.extern : Create tensor with extern function call.
    """
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    return Call(
        "int32", "tvm_call_packed", call_args, Call.Intrinsic, None, 0)


def call_pure_intrin(dtype, func_name, *args):
    """Build expression by calling a pure intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = convert(args)
    return Call(
        dtype, func_name, convert(args), Call.PureIntrinsic, None, 0)


def call_intrin(dtype, func_name, *args):
    """Build expression by calling an intrinsic function.

    Intrinsics can be overloaded with multiple data types via
    the intrinsic translation rule.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The intrinsic function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    args = convert(args)
    return Call(
        dtype, func_name, convert(args), Call.Intrinsic, None, 0)


def call_pure_extern(dtype, func_name, *args):
    """Build expression by calling a pure extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return Call(
        dtype, func_name, convert(args), Call.PureExtern, None, 0)


def call_extern(dtype, func_name, *args):
    """Build expression by calling a extern function.

    Parameters
    ----------
    dtype : str
        The data type of the result.

    func_name: str
        The extern function name.

    args : list
        Positional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return Call(
        dtype, func_name, convert(args), Call.Extern, None, 0)


def call_llvm_intrin(dtype, name, *args):
    """Build expression by calling an llvm intrinsic function

    Parameters
    ----------
    dtype : str
       The data type of the result.

    name : str
       The name of the llvm intrinsic function.

    args : list
       Poistional arguments.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.target import codegen
    llvm_id = codegen.llvm_lookup_intrinsic_id(name)
    assert llvm_id != 0, "%s is not an LLVM intrinsic" % name
    return call_pure_intrin(dtype, 'llvm_intrin', tvm.const(llvm_id, 'uint32'), *args)


@tvm._ffi.register_func("tvm.default_trace_action")
def _tvm_default_trace_action(*args):
    print(list(args))

def trace(args, trace_action="tvm.default_trace_action"):
    """Trace tensor data at the runtime.

    The trace function allows to trace specific tensor at the
    runtime. The tracing value should come as last argument.
    The trace action should be specified, by default
    tvm.default_trace_action is used.

    Parameters
    ----------
    args : list of Expr or Buffers.
        Positional arguments.

    trace_action : str.
        The name of the trace action.

    Returns
    -------
    call : PrimExpr
        The call expression.

    See Also
    --------
    tvm.tir.call_packed : Creates packed function.
    """
    if not isinstance(args, list):
        raise Exception("tvm.trace consumes the args as list type")
    call_args = [_pack_buffer(x) if isinstance(x, Buffer) else x for x in args]
    call_args.insert(0, trace_action)
    return tvm.tir.Call(
        args[-1].dtype, "tvm_call_trace_packed", call_args, tvm.tir.Call.Intrinsic, None, 0)



def min_value(dtype):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : tvm.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype)


def max_value(dtype):
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    Returns
    -------
    value : tvm.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype)


def exp(x):
    """Take exponetial of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "exp", x)


def erf(x):
    """Take gauss error function of the input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "erf", x)


def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "tanh", x)


def sigmoid(x):
    """Quick function to get sigmoid

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sigmoid", x)


def log(x):
    """Take log of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "log", x)

def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "cos", x)

def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sin", x)

def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "atan", x)

def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "sqrt", x)


def rsqrt(x):
    """Take reciprocal of square root of input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "rsqrt", x)


def floor(x):
    """Take floor of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.floor(x)


def ceil(x):
    """Take ceil of float input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.ceil(x)


def trunc(x):
    """Get truncated value of the input.

    The truncated value of the scalar x is the
    nearest integer i which is closer to zero than x is.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.trunc(x)


def abs(x):
    """Get absolute value of the input element-wise.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.abs(x)


def round(x):
    """Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.round(x)


def nearbyint(x):
    """Round elements of the array to the nearest integer.
    This intrinsic uses llvm.nearbyint instead of llvm.round
    which is faster but will results different from tvm.round.
    Notably nearbyint rounds according to the rounding mode,
    whereas tvm.round (llvm.round) ignores that.
    For differences between the two see:
    https://en.cppreference.com/w/cpp/numeric/math/round
    https://en.cppreference.com/w/cpp/numeric/math/nearbyint

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.nearbyint(x)


def isnan(x):
    """Check if input value is Nan.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _ffi_api.isnan(x)


def power(x, y):
    """x power y

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    y : PrimExpr
        The exponent

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return _ffi_api._OpPow(convert(x), convert(y))


def popcount(x):
    """Count the number of set bits in input x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "popcount", x)

def fmod(x, y):
    """Return the remainder of x divided by y with the same sign as x.

    Parameters
    ----------
    x : PrimExpr
        Input argument.
    y : PrimExpr
        Input argument.

    Returns
    -------
    z : PrimExpr
        The result.
    """
    return call_pure_intrin(x.dtype, "fmod", x, y)


def if_then_else(cond, t, f):
    """Conditional selection expression.

    Parameters
    ----------
    cond : PrimExpr
        The condition

    t : PrimExpr
        The result expression if cond is true.

    f : PrimExpr
        The result expression if cond is false.

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
    return _ffi_api._OpIfThenElse(convert(cond), convert(t), convert(f))


def div(a, b):
    """Compute a / b as in C/C++ semantics.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.
    Note
    ----
    When operands are integers, returns truncdiv(a, b).
    """
    return _ffi_api._OpDiv(a, b)


def indexdiv(a, b):
    """Compute floor(a / b) where a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexDiv(a, b)


def indexmod(a, b):
    """Compute the remainder of indexdiv. a and b are non-negative.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand, known to be non-negative.

    b : PrimExpr
        The right hand operand, known to be non-negative.

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    Use this function to split non-negative indices.
    This function may take advantage of operands'
    non-negativeness.
    """
    return _ffi_api._OpIndexMod(a, b)


def truncdiv(a, b):
    """Compute the truncdiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncDiv(a, b)


def truncmod(a, b):
    """Compute the truncmod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api._OpTruncMod(a, b)


def floordiv(a, b):
    """Compute the floordiv of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorDiv(a, b)


def floormod(a, b):
    """Compute the floormod of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api._OpFloorMod(a, b)
