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
# pylint: disable=redefined-builtin, invalid-name
"""Relax unary arithmetic operators."""
from . import _ffi_api
from ..expr import Expr
from ..utils import args_converter

###################### Arithmetic operators ######################


def abs(x: Expr) -> Expr:
    """Compute element-wise absolute value of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.abs(x)  # type: ignore


def acos(x: Expr) -> Expr:
    """Compute element-wise arc cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.acos(x)  # type: ignore


def acosh(x: Expr) -> Expr:
    """Compute element-wise arc cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.acosh(x)  # type: ignore


def asin(x: Expr) -> Expr:
    """Compute element-wise arc sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.asin(x)  # type: ignore


def asinh(x: Expr) -> Expr:
    """Compute element-wise arc sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.asinh(x)  # type: ignore


def atan(x: Expr) -> Expr:
    """Compute element-wise arc tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.atan(x)  # type: ignore


def atanh(x: Expr) -> Expr:
    """Compute element-wise arc tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.atanh(x)  # type: ignore


def bitwise_not(x: Expr) -> Expr:
    """Compute bitwise NOT of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.bitwise_not(x)  # type: ignore


def ceil(x: Expr) -> Expr:
    """Take ceil of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.ceil(x)  # type: ignore


def cos(x: Expr) -> Expr:
    """Compute element-wise cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.cos(x)  # type: ignore


def cosh(x: Expr) -> Expr:
    """Compute element-wise cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.cosh(x)  # type: ignore


def exp(x: Expr) -> Expr:
    """Compute element-wise exp of data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.exp(x)  # type: ignore


def floor(x: Expr) -> Expr:
    """Take floor of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.floor(x)  # type: ignore


def log(x: Expr) -> Expr:
    """Compute element-wise natural logarithm of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.log(x)  # type: ignore


def logical_not(x: Expr) -> Expr:
    """Compute logical NOT of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.logical_not(x)  # type: ignore


def negative(x: Expr) -> Expr:
    """Compute element-wise negative of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result
    """
    return _ffi_api.negative(x)  # type: ignore


def round(x: Expr) -> Expr:
    """Rounds each element of the input data to nearest integer.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.round(x)  # type: ignore


def rsqrt(x: Expr) -> Expr:
    """Compute element-wise reciprocal square root of the input data.

    .. math::

      1/sqrt(x)

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.rsqrt(x)  # type: ignore


def sigmoid(x: Expr) -> Expr:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sigmoid(x)  # type: ignore


def sign(x: Expr) -> Expr:
    """Returns an indication of the sign of a number for each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sign(x)  # type: ignore


def sin(x: Expr) -> Expr:
    """Compute element-wise sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sin(x)  # type: ignore


def sinh(x: Expr) -> Expr:
    """Compute element-wise sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sinh(x)  # type: ignore


def square(x: Expr) -> Expr:
    """Squares each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.square(x)  # type: ignore


def sqrt(x: Expr) -> Expr:
    """Compute element-wise square root of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sqrt(x)  # type: ignore


def tan(x: Expr) -> Expr:
    """Compute element-wise tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.tan(x)  # type: ignore


def tanh(x: Expr) -> Expr:
    """Compute element-wise tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.tanh(x)  # type: ignore


@args_converter.auto
def clip(x: Expr, min: Expr, max: Expr) -> Expr:
    """Clips tensor values to a specified min and max.

    Parameters
    ----------
    x : relax.Expr
        The input data

    min : relax.Expr
        The minimum value

    max : relax.Expr
        The maximum value

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.clip(x, min, max)  # type: ignore


def erf(x: Expr) -> Expr:
    """Computes the error function of the input.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        Computed error function for each element.
    """
    return _ffi_api.erf(x)  # type: ignore


###################### Check operators ######################


def isfinite(x: Expr) -> Expr:
    """Check if input value is finite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isfinite(x)  # type: ignore


def isinf(x: Expr) -> Expr:
    """Check if input value is infinite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isinf(x)  # type: ignore


def isnan(x: Expr) -> Expr:
    """Check if input value is Nan.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isnan(x)  # type: ignore
