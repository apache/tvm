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
"""Splittable and parallelizable PRNG kernels."""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import

import sys
import numpy as np

from ...expr import Constant
from .... import nd
from . import _make


def threefry_key(seed):
    """Create a new Threefry random number generator key.

    Example
    -------

    .. code-block:: python

        gen = threefry_key(0)
        _, random_number = threefry_generate(gen, (4,))

    Parameters
    ----------
    seed : int
        Starting seed for the key

    Returns
    -------
    key : relay.Expr
        New key to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.
    """
    s = np.frombuffer(seed.to_bytes(32, sys.byteorder), dtype="uint64")
    a = np.concatenate((s, np.array([0, 0, 0, 0, 1 << 63, 0], dtype="uint64")))
    return Constant(nd.array(a))


def threefry_generate(key, shape):
    """Generate an array of random bits (`uint64`) using the Threefry algorithm

    Example
    -------

    .. code-block:: python

        key = threefry_key(0)
        new_key, random1 = threefry_generate(key, (4,))
        _, random2 = threefry_generate(new_key, (4,))
        # random1 and random2 are different random numbers

    Parameters
    ----------
    key : relay.Expr
        key that uniquely determines the random values. Multiple uses with the
        same key will generate the same random values. This key should be
        treated as an opaque pointer. You can create one from calling
        :py:func:`threefry_key`, :py:func:`threefry_split`, or
        :py:func:`threefry_generate`. **Do not use this key again after calling
        this function.**

    shape : Sequence[int]
        Desired outputs shape of random numbers. **Currently the total
        number of elements must be a multiple of 4.**

    Returns
    -------
    new_key : relay.Expr
        New key to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.

    random_array : relay.Expr
        Array of random numbers. Has shape `shape`.
    """
    return _make.threefry_generate(key, shape)


def threefry_split(key):
    """Split an existing Threefry key into two new ones.

    This is useful if you have to subsequent calls which each need their own
    independent random number generation.

    Example
    -------

    .. code-block:: python

        def foo(key):
            new_key, num = threefry_generate(key, (4,))
            return num

        key = threefry_key(0)
        key1, key2 = threefry_split(key)
        assert foo(key1) != foo(key2)

    Parameters
    ----------
    key : relay.Expr
        key that uniquely determines the random values. Multiple uses with the
        same generator will generate the same random values. This generator should be
        treated as an opaque pointer. You can create one from calling
        :py:func:`threefry_key`, :py:func:`threefry_split`, or
        :py:func:`threefry_generate`. **Do not use this generator again after calling
        this function.**

    Returns
    -------
    new_key_1 : relay.Expr
        New key to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.

    new_key_2 : relay.Expr
        New key to pass to future uses of :py:func:`threefry_split` or
        :py:func:`threefry_generate`.
    """
    return _make.threefry_split(key)
