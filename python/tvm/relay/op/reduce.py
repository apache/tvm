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
"""Reduce operators."""
# pylint: disable=redefined-builtin

from . import _make

def argmax(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmax(data, axis, keepdims, exclude)

def argmin(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmin(data, axis, keepdims, exclude)


def sum(data, axis=None, keepdims=False, exclude=False):
    """Computes the sum of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.sum(data, axis, keepdims, exclude)


def all(data, axis=None, keepdims=False, exclude=False):
    """Computes the logical AND of boolean array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input boolean tensor

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all of the elements of the input array. If axis is
        negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

    data = relay.Constant(tvm.nd.array([[[ True,  True,  True],
                                         [ True,  True,  True],
                                         [False,  True, False]],
                                        [[ True, False, False],
                                         [ True,  True, False],
                                         [False,  True,  True]]]))

    relay.all(data, axis=1)
    # [[False,  True, False],
    # [False, False, False]]

    relay.all(data, axis=0)
    # [[ True, False, False],
    # [ True,  True, False],
    # [False,  True, False]]

    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.all(data, axis, keepdims, exclude)


def max(data, axis=None, keepdims=False, exclude=False):
    """ Computes the max of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which the max operation is performed.
        The default, axis=None, will find the max element from all of the elements of the input
        array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.max(data, axis, keepdims, exclude)


def min(data, axis=None, keepdims=False, exclude=False):
    """Computes the min of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a minimum operation is performed.
        The default, axis=None, will find the minimum element from all
        of the elements of the input array. If axis is negative it counts from
        the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.min(data, axis, keepdims, exclude)


def mean(data, axis=None, keepdims=False, exclude=False):
    """Computes the mean of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.mean(data, axis, keepdims, exclude)


def prod(data, axis=None, keepdims=False, exclude=False):
    """Computes the products of array elements over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a product is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.prod(data, axis, keepdims, exclude)
