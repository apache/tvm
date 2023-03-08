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
from tvm.ir import Span

from . import _ffi_api
from ..expr import Expr
from ..utils import args_converter


###################### Arithmetic operators ######################


def abs(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise absolute value of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.abs(x, span)  # type: ignore


def acos(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.acos(x, span)  # type: ignore


def acosh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.acosh(x, span)  # type: ignore


def asin(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.asin(x, span)  # type: ignore


def asinh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.asinh(x, span)  # type: ignore


def atan(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.atan(x, span)  # type: ignore


def atanh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise arc tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.atanh(x, span)  # type: ignore


def ceil(x: Expr, span: Span = None) -> Expr:
    """Take ceil of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.ceil(x, span)  # type: ignore


def cos(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.cos(x, span)  # type: ignore


def cosh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.cosh(x, span)  # type: ignore


def exp(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise exp of data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.exp(x, span)  # type: ignore


def floor(x: Expr, span: Span = None) -> Expr:
    """Take floor of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.floor(x, span)  # type: ignore


def log(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise natural logarithm of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.log(x, span)  # type: ignore


def negative(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise negative of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result
    """
    return _ffi_api.negative(x, span)  # type: ignore


def round(x: Expr, span: Span = None) -> Expr:
    """Rounds each element of the input data to nearest integer.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.round(x, span)  # type: ignore


def sigmoid(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sigmoid(x, span)  # type: ignore


def sign(x: Expr, span: Span = None) -> Expr:
    """Returns an indication of the sign of a number for each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.sign(x, span)  # type: ignore


def sin(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sin(x, span)  # type: ignore


def sinh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sinh(x, span)  # type: ignore


def square(x: Expr, span: Span = None) -> Expr:
    """Squares each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.square(x, span)  # type: ignore


def sqrt(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise square root of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.sqrt(x, span)  # type: ignore


def tan(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.tan(x, span)  # type: ignore


def tanh(x: Expr, span: Span = None) -> Expr:
    """Compute element-wise tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.tanh(x, span)  # type: ignore


@args_converter.auto
def clip(x: Expr, min: Expr, max: Expr, span: Span = None) -> Expr:
    """Clips tensor values to a specified min and max.

    Parameters
    ----------
    x : relax.Expr
        The input data

    min : relax.Expr
        The minimum value

    max : relax.Expr
        The maximum value

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.clip(x, min, max, span)  # type: ignore


###################### Check operators ######################


def isfinite(x: Expr, span: Span = None) -> Expr:
    """Check if input value is finite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isfinite(x, span)  # type: ignore


def isinf(x: Expr, span: Span = None) -> Expr:
    """Check if input value is infinite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isinf(x, span)  # type: ignore


def isnan(x: Expr, span: Span = None) -> Expr:
    """Check if input value is Nan.

    Parameters
    ----------
    x : relax.Expr
        The input data

    span : Span
        The source code span.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.isnan(x, span)  # type: ignore
