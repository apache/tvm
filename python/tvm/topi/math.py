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
"""Elementwise operators"""
# pylint: disable=redefined-builtin,unused-argument
import tvm
from tvm import te
from . import tag
from . import cpp
from .utils import get_const_tuple


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def identity(x):
    """Take identity of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return te.compute(x.shape, lambda *i: x(*i))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def negative(x):
    """Take negation of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return te.compute(x.shape, lambda *i: -x(*i))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.exp(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def erf(x):
    """Take gauss error function of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.erf(x(*i)))


@tvm.target.generic_func
def erf_legalize(attrs, inputs, types):
    """Legalizes ERF op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr.
    """
    # Note changed by default.
    return None


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.tanh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def tan(x):
    """Take tan of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.tan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.cos(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def cosh(x):
    """Take cosh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.cosh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sin(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sinh(x):
    """Take sinh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sinh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def acos(x):
    """Take arc cos of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.acos(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def acosh(x):
    """Take arc cosh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.acosh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def asin(x):
    """Take arc sin of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.asin(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def asinh(x):
    """Take arc sinh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.asinh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.atan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def atanh(x):
    """Take atanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.atanh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def floor(x):
    """Take floor of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.floor(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def ceil(x):
    """Take ceil of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.ceil(x(*i)))


def sign(x):
    """Returns -1, 0, 1 based on sign of x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.sign(x)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def trunc(x):
    """Take truncated value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.trunc(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def abs(x):
    """Take absolute value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.abs(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isnan(x):
    """Check if value of x is NaN, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isnan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isfinite(x):
    """Check if value of x is finite, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isfinite(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isinf(x):
    """Check if value of x is infinite, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isinf(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def round(x):
    """Round elements of x to nearest integer.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.round(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def log(x):
    """Take logarithm of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.log(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def log2(x):
    """Take logarithm to the base 2 of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.log2(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def log10(x):
    """Take logarithm to the base 10 of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.log10(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sqrt(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def rsqrt(x):
    """Take inverse square root of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.rsqrt(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sigmoid(x):
    """Take sigmoid tanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sigmoid(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def left_shift(x, n):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: x(*i) << n)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def right_shift(x, n):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: x(*i) >> n)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def clip(x, a_min, a_max):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    a_min : int or float
        Minimum value.
    a_max : int or float
        Maximum value.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        value = x(*indices)
        const_min = tvm.tir.const(a_min, value.dtype)
        const_max = tvm.tir.const(a_max, value.dtype)
        return tvm.te.max(tvm.te.min(value, const_max), const_min)

    return te.compute(x.shape, _compute)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def fixed_point_multiply(x, multiplier, shift):
    """Fixed point multiplication between data and a fixed point
    constant expressed as multiplier * 2^(-shift), where multiplier
    is a Q-number with 31 fractional bits

    Parameters
    ----------
    x : tvm.te.Tensor or Expr
        Input argument.
    multiplier : int
        Multiplier of a fixed floating point number described as multiplier*2^(-shift).
    shift : int
        Shift of a fixed floating point number described as multiplier*2^(-shift).

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        value = x(*indices)
        return tvm.tir.q_multiply_shift(
            value,
            tvm.tir.const(multiplier, "int32"),
            tvm.tir.const(31, "int32"),
            tvm.tir.const(shift, "int32"),
        )

    return te.compute(x.shape, _compute)


@tvm.te.tag_scope(tag=tag.BROADCAST)
def fixed_point_multiply_per_axis(
    x: te.Tensor,
    y: te.Tensor,
    lshift: te.Tensor,
    rshift: te.Tensor,
    is_lshift_required: int,
    is_rshift_required: int,
    axes,
):
    """Fixed point multiplication between data and a fixed point constant expressed as
    multiplier * 2^(-shift), where multiplier is a Q-number with 31 fractional bits

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    y : tvm.te.Tensor
        Multiplier of a fixed floating point number described as multiplier*2^(-shift).
    lshift : tvm.te.Tensor
        Left shifts of a fixed floating point number described as multiplier*2^(-shift).
    rshift : tvm.te.Tensor
        Right shifts of a fixed floating point number described as multiplier*2^(-shift).
    is_lshift_required : int
        Whether we need to do left shift or not.
    is_rshift_required : int
        Whether we need to do right shift or not.

    Returns
    -------
    z : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        elements = []
        for element in get_const_tuple(axes):
            elements += [indices[element]]
        param_indices = tuple(elements)

        value = x(*indices)
        m = y(*param_indices)
        l_shift = lshift(*param_indices)
        r_shift = rshift(*param_indices)
        return tvm.tir.q_multiply_shift_per_axis(
            value,
            m,
            l_shift,
            r_shift,
            tvm.tir.const(31, "int32"),
            tvm.tir.const(is_lshift_required, "bool"),
            tvm.tir.const(is_rshift_required, "bool"),
        )

    return te.compute(x.shape, _compute)


def cast(x, dtype, span=None):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.te.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    span : Optional[Span]
        The location of the cast in the source.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if isinstance(x, te.tensor.Tensor):
        return te.compute(x.shape, lambda *i: x(*i).astype(dtype), tag=tag.ELEMWISE)
    # pylint: disable=import-outside-toplevel
    from tvm.tir import _ffi_api

    return _ffi_api._cast(dtype, x, span)


def reinterpret(x, dtype):
    """Reinterpret input to specified data type.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.reinterpret(x, dtype)


def fast_exp(x):
    """Take exponential of input x using fast_exp implementation

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.fast_exp(x, x.dtype, tag.ELEMWISE)


def fast_tanh(x):
    """Take hyperbolic tangent of input x using fast_tanh implementation

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.fast_tanh(x, x.dtype, tag.ELEMWISE)


def fast_erf(x):
    """Take gauss error function of input x using fast_erf implementation.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.fast_erf(x, x.dtype, tag.ELEMWISE)


def ceil_log2(x):
    """Compute integer ceil log2 with a special code path for vulkan
    SPIR-V does not support log2 on fp64. Instead, we compute integer ceil_log2 via clz
    intrinsic when the target is vulkan.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if not isinstance(x, tvm.tir.PrimExpr):
        x = tvm.tir.const(x)

    if "float" in x.dtype:
        return tvm.tir.ceil(tvm.tir.log2(x))

    if "vulkan" in tvm.target.Target.current().kind.name:
        clz = tvm.tir.clz(x)
        bits = int(x.dtype[-2:])
        res = tvm.tir.if_then_else(x & (x - 1) == 0, bits - clz - 1, bits - clz)

        if res.dtype != x.dtype:
            return cast(res, x.dtype)

        return res

    return cast(tvm.tir.ceil(tvm.tir.log2(cast(x, "float64"))), x.dtype)
