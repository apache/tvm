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
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.sum(data, axis, keepdims)


def max(data, axis=None, keepdims=False):
    """Maximum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.max(data, axis, keepdims)


def min(data, axis=None, keepdims=False):
    """Minimum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.min(data, axis, keepdims)


def argmax(data, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.argmax(data, axis, keepdims)


def argmin(data, axis=None, keepdims=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.argmin(data, axis, keepdims)


def prod(data, axis=None, keepdims=False):
    """Product of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
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
    ret : tvm.Tensor
    """
    return cpp.prod(data, axis, keepdims)
