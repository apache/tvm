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
"""Relax binary arithmetic and comparison operators."""
from . import _ffi_api
from ..expr import Expr

###################### Arithmetic operators ######################


def add(x1: Expr, x2: Expr) -> Expr:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.

    Examples
    --------
    .. code:: python

      bb = relax.BlockBuilder()
      a = relax.Var("a", relax.TensorStructInfo(shape=(2, 3), dtype="float32"))
      b = relax.Var("b", relax.TensorStructInfo(shape=(2, 1), dtype="float32"))
      c = bb.normalize(relax.op.add(a, b))  # c has TensorStructInfo(shape=(2, 3), dtype="float32")
    """
    return _ffi_api.add(x1, x2)  # type: ignore


def divide(x1: Expr, x2: Expr) -> Expr:
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.divide(x1, x2)  # type: ignore


def floor_divide(x1: Expr, x2: Expr) -> Expr:
    """Floor division with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.floor_divide(x1, x2)  # type: ignore


def multiply(x1: Expr, x2: Expr) -> Expr:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _ffi_api.multiply(x1, x2)  # type: ignore


def power(x1: Expr, x2: Expr):
    """Power with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.power(x1, x2)  # type: ignore


def subtract(x1: Expr, x2: Expr) -> Expr:
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.subtract(x1, x2)  # type: ignore


def mod(x1: Expr, x2: Expr) -> Expr:
    """Modulo with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.
    """
    return _ffi_api.mod(x1, x2)  # type: ignore


def floor_mod(x1: Expr, x2: Expr) -> Expr:
    """Floor modulo with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.
    """
    return _ffi_api.floor_mod(x1, x2)  # type: ignore


###################### Comparison operators ######################


def equal(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs == rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.equal(x1, x2)  # type: ignore


def greater(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs > rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.greater(x1, x2)  # type: ignore


def greater_equal(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs >= rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.greater_equal(x1, x2)  # type: ignore


def less(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs < rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.less(x1, x2)  # type: ignore


def less_equal(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs <= rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.less_equal(x1, x2)  # type: ignore


def not_equal(x1: Expr, x2: Expr) -> Expr:
    """Broadcasted element-wise test for (lhs != rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.not_equal(x1, x2)  # type: ignore


def maximum(x1: Expr, x2: Expr) -> Expr:
    """Element-wise maximum

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.maximum(x1, x2)


def minimum(x1: Expr, x2: Expr) -> Expr:
    """Element-wise minimum

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.minimum(x1, x2)


###################### Logical operators ######################


def logical_and(x1: Expr, x2: Expr) -> Expr:
    """Logical AND
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.logical_and(x1, x2)


def logical_or(x1: Expr, x2: Expr) -> Expr:
    """Logical OR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.logical_or(x1, x2)


def logical_xor(x1: Expr, x2: Expr) -> Expr:
    """Logical XOR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.logical_xor(x1, x2)


###################### Bitwise operators ######################


def bitwise_and(x1: Expr, x2: Expr) -> Expr:
    """Bitwise AND
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.bitwise_and(x1, x2)


def bitwise_or(x1: Expr, x2: Expr) -> Expr:
    """Bitwise OR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.bitwise_or(x1, x2)


def bitwise_xor(x1: Expr, x2: Expr) -> Expr:
    """Bitwise XOR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.bitwise_xor(x1, x2)


def left_shift(x1: Expr, x2: Expr) -> Expr:
    """Bitwise Shift Left
    Parameters
    ----------
    x1 : relax.Expr
        The input tensor to be shifted.
    x2 : relax.Expr
        The number of positions to shift.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.left_shift(x1, x2)


def right_shift(x1: Expr, x2: Expr) -> Expr:
    """Bitwise Shift Right
    Parameters
    ----------
    x1 : relax.Expr
        The input tensor to be shifted.
    x2 : relax.Expr
        The number of positions to shift.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.right_shift(x1, x2)
