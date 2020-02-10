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
(e.g., tvm.generic.multitply ~ numpy.multiply).
The default implementation is used by tvm.ExprOp.
"""
# pylint: disable=unused-argument
from . import make as _make

#Operator precedence used when overloading.
__op_priority__ = 0


def add(lhs, rhs):
    """Generic add operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of add operaton.
    """
    return _make._OpAdd(lhs, rhs)


def subtract(lhs, rhs):
    """Generic subtract operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of subtract operaton.
    """
    return _make._OpSub(lhs, rhs)


def multiply(lhs, rhs):
    """Generic multiply operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of multiply operaton.
    """
    return _make._OpMul(lhs, rhs)

def divide(lhs, rhs):
    """Generic divide operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make._OpDiv(lhs, rhs)

def floordiv(lhs, rhs):
    """Generic floordiv operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make._OpFloorDiv(lhs, rhs)


def cast(src, dtype):
    """Generic cast operator.

    Parameters
    ----------
    src : object
        The source operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make._cast(dtype, src)
