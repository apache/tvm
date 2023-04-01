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
# pylint: disable=redefined-builtin
"""The basic Relax operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


## (TVM-TOOL) py_op begin elementwise/*
def abs(
    a: ty.Tensor,
) -> Call:
    """Elementwise abs

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.abs")
    return _ffi_func(a)


def acos(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise acos

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.acos")
    return _ffi_func(a)


def acosh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise acosh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.acosh")
    return _ffi_func(a)


def add(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise add

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.add")
    return _ffi_func(a, b)


def asin(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise asin

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.asin")
    return _ffi_func(a)


def asinh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise asinh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.asinh")
    return _ffi_func(a)


def atan(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise atan

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.atan")
    return _ffi_func(a)


def atan2(
    a: ty.FloatTensor,
    b: ty.FloatTensor,
) -> Call:
    """Elementwise atan2

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc
    b : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    b = tg.check(1, "b", tg.FloatTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.atan2")
    return _ffi_func(a, b)


def atanh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise atanh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.atanh")
    return _ffi_func(a)


def bitwise_and(
    a: ty.IntTensor,
    b: ty.IntTensor,
) -> Call:
    """Elementwise bitwise and

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc
    b : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    b = tg.check(1, "b", tg.IntTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_and")
    return _ffi_func(a, b)


def bitwise_invert(
    a: ty.IntTensor,
) -> Call:
    """Elementwise bitwise invert

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_invert")
    return _ffi_func(a)


def bitwise_left_shift(
    a: ty.IntTensor,
    b: ty.IntTensor,
) -> Call:
    """Elementwise bitwise left shift

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc
    b : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    b = tg.check(1, "b", tg.IntTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_left_shift")
    return _ffi_func(a, b)


def bitwise_or(
    a: ty.IntTensor,
    b: ty.IntTensor,
) -> Call:
    """Elementwise bitwise or

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc
    b : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    b = tg.check(1, "b", tg.IntTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_or")
    return _ffi_func(a, b)


def bitwise_right_shift(
    a: ty.IntTensor,
    b: ty.IntTensor,
) -> Call:
    """Elementwise bitwise right shift

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc
    b : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    b = tg.check(1, "b", tg.IntTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_right_shift")
    return _ffi_func(a, b)


def bitwise_xor(
    a: ty.IntTensor,
    b: ty.IntTensor,
) -> Call:
    """Elementwise bitwise xor

    Parameters
    ----------
    a : ty.IntTensor
        TODO(tvm-unity-team): add doc
    b : ty.IntTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.IntTensor([]), a)
    b = tg.check(1, "b", tg.IntTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.bitwise_xor")
    return _ffi_func(a, b)


def ceil(
    a: ty.Tensor,
) -> Call:
    """Elementwise ceil

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.ceil")
    return _ffi_func(a)


def cos(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise cos

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.cos")
    return _ffi_func(a)


def cosh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise cosh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.cosh")
    return _ffi_func(a)


def divide(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise divide

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.divide")
    return _ffi_func(a, b)


def equal(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise equal

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.equal")
    return _ffi_func(a, b)


def exp(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise exp

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.exp")
    return _ffi_func(a)


def floor(
    a: ty.Tensor,
) -> Call:
    """Elementwise floor

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.floor")
    return _ffi_func(a)


def floor_divide(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise floor divide

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.floor_divide")
    return _ffi_func(a, b)


def greater(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise greater

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.greater")
    return _ffi_func(a, b)


def greater_equal(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise greater equal

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.greater_equal")
    return _ffi_func(a, b)


def isfinite(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise isfinite

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.isfinite")
    return _ffi_func(a)


def isinf(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise isinf

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.isinf")
    return _ffi_func(a)


def isnan(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise isnan

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.isnan")
    return _ffi_func(a)


def less(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise less

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.less")
    return _ffi_func(a, b)


def less_equal(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise less equal

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.less_equal")
    return _ffi_func(a, b)


def log(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise log

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.log")
    return _ffi_func(a)


def log10(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise log10

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.log10")
    return _ffi_func(a)


def log1p(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise log1p

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.log1p")
    return _ffi_func(a)


def log2(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise log2

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.log2")
    return _ffi_func(a)


def logical_and(
    a: ty.BoolTensor,
    b: ty.BoolTensor,
) -> Call:
    """Elementwise logical and

    Parameters
    ----------
    a : ty.BoolTensor
        TODO(tvm-unity-team): add doc
    b : ty.BoolTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.BoolTensor([]), a)
    b = tg.check(1, "b", tg.BoolTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.logical_and")
    return _ffi_func(a, b)


def logical_not(
    a: ty.BoolTensor,
) -> Call:
    """Elementwise logical not

    Parameters
    ----------
    a : ty.BoolTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.BoolTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.logical_not")
    return _ffi_func(a)


def logical_or(
    a: ty.BoolTensor,
    b: ty.BoolTensor,
) -> Call:
    """Elementwise logical or

    Parameters
    ----------
    a : ty.BoolTensor
        TODO(tvm-unity-team): add doc
    b : ty.BoolTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.BoolTensor([]), a)
    b = tg.check(1, "b", tg.BoolTensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.logical_or")
    return _ffi_func(a, b)


def multiply(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise multiply

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.multiply")
    return _ffi_func(a, b)


def negative(
    a: ty.Tensor,
) -> Call:
    """Elementwise negative

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.negative")
    return _ffi_func(a)


def not_equal(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise not equal

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.not_equal")
    return _ffi_func(a, b)


def positive(
    a: ty.Tensor,
) -> Call:
    """Elementwise positive

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.positive")
    return _ffi_func(a)


def pow(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise pow

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.pow")
    return _ffi_func(a, b)


def power(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise pow

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.power")
    return _ffi_func(a, b)


def remainder(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise remainder

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.remainder")
    return _ffi_func(a, b)


def round(
    a: ty.Tensor,
) -> Call:
    """Elementwise round

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.round")
    return _ffi_func(a)


def sin(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise sin

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.sin")
    return _ffi_func(a)


def sinh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise sinh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.sinh")
    return _ffi_func(a)


def sqrt(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise sqrt

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.sqrt")
    return _ffi_func(a)


def square(
    a: ty.Tensor,
) -> Call:
    """Elementwise square

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.square")
    return _ffi_func(a)


def subtract(
    a: ty.Tensor,
    b: ty.Tensor,
) -> Call:
    """Elementwise subtract

    Parameters
    ----------
    a : ty.Tensor
        TODO(tvm-unity-team): add doc
    b : ty.Tensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.Tensor([]), a)
    b = tg.check(1, "b", tg.Tensor([]), b)
    _ffi_func = _ffi.get_global_func("relax.op.subtract")
    return _ffi_func(a, b)


def tan(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise tan

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.tan")
    return _ffi_func(a)


def tanh(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise tanh

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.tanh")
    return _ffi_func(a)


def trunc(
    a: ty.FloatTensor,
) -> Call:
    """Elementwise trunc

    Parameters
    ----------
    a : ty.FloatTensor
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    a = tg.check(0, "a", tg.FloatTensor([]), a)
    _ffi_func = _ffi.get_global_func("relax.op.trunc")
    return _ffi_func(a)


## (TVM-TOOL) py_op end elementwise/*
