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
"""Classic algorithm operation"""
from __future__ import absolute_import as _abs

import sys

import numpy as np

from ... import nd
from ..expr import Constant, Expr, TupleWrapper
from . import _make
from .dyn import _make as _dyn_make


def argsort(data, axis=-1, is_ascend=1, dtype="int32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    valid_count : tvm.te.Tensor
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the output indices.

    Returns
    -------
    out : relay.Expr
        Tensor with same shape as data.
    """
    return _make.argsort(data, axis, is_ascend, dtype)


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int32"):
    """Get the top k elements in an input tensor along the given axis.

    ret_type specifies the return type, can be one of ("both", "values", "indices").

    Parameters
    ----------
    data : relay.Expr
        The input data tensor.

    k : int or relay.Expr, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : relay.Expr or List[relay.Expr]
        The computed result.
    """
    if isinstance(k, Constant):
        k = k.data.asnumpy().item()
    if isinstance(k, Expr):
        out = _dyn_make.topk(data, k, axis, ret_type, is_ascend, dtype)
    else:
        out = _make.topk(data, k, axis, ret_type, is_ascend, dtype)
    if ret_type == "both":
        return TupleWrapper(out, 2)
    return out


def threefry_seed(seed):
    """Create a new Threefry random number generator.

    Example
    -------

    .. code-block:: python

        gen = threefry_seed(0)
        _, random_number = threefry_generate(gen, (1,))

    Parameters
    ----------
    seed : int
        Starting seed for the generator

    Returns
    -------
    gen : relay.Expr
        New generator to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.
    """
    s = np.frombuffer(seed.to_bytes(32, sys.byteorder), dtype="uint64")
    a = np.concatenate((s, np.array([0, 0, 0, 0, 1 << 63, 0], dtype="uint64")))
    return Constant(nd.array(a))


def threefry_generate(gen, shape):
    """Generate an array of random numbers using the Threefry algorithm

    Example
    -------

    .. code-block:: python

        gen = threefry_seed(0)
        new_gen, random1 = threefry_generate(gen, (1,))
        _, random2 = threefry_generate(new_gen, (1,))
        # random1 and random2 are different random numbers

    Parameters
    ----------
    gen : relay.Expr
        generator that uniquely determines the random values. Multiple uses with the
        same generator will generate the same random values. This generator should be
        treated as an opaque pointer. You can create one from calling
        :py:func:`threefry_seed`, :py:func:`threefry_split`, or
        :py:func:`threefry_generate`. **Do not use this generator again after calling
        this function.**

    shape : Sequence[int]
        Desired outputs shape of random numbers

    Returns
    -------
    new_gen : relay.Expr
        New generator to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.

    random_array : relay.Expr
        Array of random numbers. Has shape `shape`.
    """
    return _make.threefry_generate(gen, shape)


def threefry_split(gen):
    """Split an existing threefry generator into two new ones.

    This is useful if you have to subsequent calls which each need their own
    random number generation.

    Example
    -------

    .. code-block:: python

        def foo(gen):
            new_gen, num = threefry_generate(gen, (1,))
            return num

        gen = threefry_seed(0)
        gen1, gen2 = threefry_split(gen)
        assert foo(gen1) != foo(gen2)

    Parameters
    ----------
    gen : relay.Expr
        generator that uniquely determines the random values. Multiple uses with the
        same generator will generate the same random values. This generator should be
        treated as an opaque pointer. You can create one from calling
        :py:func:`threefry_seed`, :py:func:`threefry_split`, or
        :py:func:`threefry_generate`. **Do not use this generator again after calling
        this function.**

    Returns
    -------
    new_gen_1 : relay.Expr
        New generator to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.

    new_gen_2 : relay.Expr
        New generator to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.
    """
    return _make.threefry_split(gen)
