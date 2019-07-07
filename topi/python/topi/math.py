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
# pylint: disable=redefined-builtin
from __future__ import absolute_import as _abs
import tvm
from . import tag
from . import cpp

@tvm.tag_scope(tag=tag.ELEMWISE)
def identity(x):
    """Take identity of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: x(*i))


@tvm.tag_scope(tag=tag.ELEMWISE)
def negative(x):
    """Take negation of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return tvm.compute(x.shape, lambda *i: -x(*i))


@tvm.tag_scope(tag=tag.ELEMWISE)
def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.exp(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.tanh(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def floor(x):
    """Take floor of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.floor(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def ceil(x):
    """Take ceil of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.ceil(x(*i)))

def sign(x):
    """Returns -1, 0, 1 based on sign of x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return cpp.sign(x)

@tvm.tag_scope(tag=tag.ELEMWISE)
def trunc(x):
    """Take truncated value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.trunc(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def abs(x):
    """Take absolute value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.abs(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def round(x):
    """Round elements of x to nearest integer.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.round(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def log(x):
    """Take logarithm of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.log(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sqrt(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def rsqrt(x):
    """Take inverse square root of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.rsqrt(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def sigmoid(x):
    """Take sigmoid tanh of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: tvm.sigmoid(x(*i)))


@tvm.tag_scope(tag=tag.ELEMWISE)
def left_shift(x, n):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) << n)


@tvm.tag_scope(tag=tag.ELEMWISE)
def right_shift(x, n):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(x.shape, lambda *i: x(*i) >> n)


@tvm.tag_scope(tag=tag.ELEMWISE)
def clip(x, a_min, a_max):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    a_min : int or float
        Minimum value.
    a_max : int or float
        Maximum value.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    def _compute(*indices):
        value = x(*indices)
        const_min = tvm.const(a_min, value.dtype)
        const_max = tvm.const(a_max, value.dtype)
        return tvm.max(tvm.min(value, const_max), const_min)
    return tvm.compute(x.shape, _compute)


def cast(x, dtype):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    if isinstance(x, tvm.tensor.Tensor):
        return tvm.compute(
            x.shape, lambda *i: x(*i).astype(dtype), tag=tag.ELEMWISE)
    return tvm.make._cast(dtype, x)
