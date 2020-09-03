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
# pylint: disable=redefined-builtin,consider-using-enumerate,no-member
"""Reduce operators"""
from __future__ import absolute_import as _abs
from . import cpp

def _get_real_axis(ndim, axis):
    if axis is None:
        real_axis = list(range(ndim))
    else:
        if isinstance(axis, int):
            axis = [axis]
        else:
            assert isinstance(axis, (list, tuple))
        real_axis = []
        for ele in axis:
            if ele < 0:
                ele += ndim
            if ele >= ndim:
                raise ValueError(
                    "{} exceeds the maximum dimension {}. Received axis={}".format(ele, ndim, axis))
            real_axis.append(ele)
        real_axis.sort()
        real_axis = list(set(real_axis))  # Remove the duplicates
    return real_axis


def sum(data, axis=None, keepdims=False):
    """Sum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.sum(data, axis, keepdims)


def all(data, axis=None, keepdims=False):
    """Logical AND of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm boolean tensor

    axis : None or int or tuple of int
        Axis or axes along which a logical AND is performed.
        The default, axis=None, will perform logical AND over all elements of the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.all(data, axis, keepdims)


def any(data, axis=None, keepdims=False):
    """Logical OR of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm boolean tensor

    axis : None or int or tuple of int
        Axis or axes along which a logical OR is performed.
        The default, axis=None, will perform logical OR over all elements of the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.any(data, axis, keepdims)


def max(data, axis=None, keepdims=False):
    """Maximum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which the max operation is performed.
        The default, axis=None, will find the max element from all of the elements of the input
        array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.max(data, axis, keepdims)


def min(data, axis=None, keepdims=False):
    """Minimum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a minimum operation is performed.
        The default, axis=None, will find the minimum element from all of the elements of the
        input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.min(data, axis, keepdims)


def argmax(data, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.argmax(data, axis, keepdims)


def argmin(data, axis=None, keepdims=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.argmin(data, axis, keepdims)


def prod(data, axis=None, keepdims=False):
    """Product of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a prod operation is performed.
        The default, axis=None, will get the prod element over all of the elements of the
        input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.prod(data, axis, keepdims)
