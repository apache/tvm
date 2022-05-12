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
"""Generic opertors in TVM.
We follow the numpy naming convention for this interface
(e.g., tvm.tir.generic.multitply ~ numpy.multiply).
The default implementation is used by tvm.ExprOp.
"""
# pylint: disable=unused-argument
from . import _ffi_api

# Operator precedence used when overloading.
__op_priority__ = 0


def add(lhs, rhs, span=None):
    """Generic add operator.

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
        The result Expr of add operaton.
    """
    return _ffi_api._OpAdd(lhs, rhs, span)  # type: ignore


def subtract(lhs, rhs, span=None):
    """Generic subtract operator.

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
        The result Expr of subtract operaton.
    """
    return _ffi_api._OpSub(lhs, rhs, span)  # type: ignore


def multiply(lhs, rhs, span=None):
    """Generic multiply operator.

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
        The result Expr of multiply operaton.
    """
    return _ffi_api._OpMul(lhs, rhs, span)  # type: ignore


def divide(lhs, rhs, span=None):
    """Generic divide operator.

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
        The result Expr of divide operaton.
    """
    return _ffi_api._OpDiv(lhs, rhs, span)  # type: ignore


def floordiv(lhs, rhs, span=None):
    """Generic floordiv operator.

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
        The result Expr of floordiv operaton.
    """
    return _ffi_api._OpFloorDiv(lhs, rhs, span)  # type: ignore


def cast(src, dtype, span=None):
    """Generic cast operator.

    Parameters
    ----------
    src : object
        The source operand.
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.Expr
        The result Expr of cast operaton.
    """
    return _ffi_api._cast(dtype, src, span)  # type: ignore
