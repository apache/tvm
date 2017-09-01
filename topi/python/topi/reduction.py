# pylint: disable=redefined-builtin,consider-using-enumerate
"""Reduce operators"""
from __future__ import absolute_import as _abs
import tvm


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


def get_reduce_out_shape(src_shape, axis=None, keepdims=False):
    real_axis = _get_real_axis(len(src_shape), axis)
    if keepdims:
        dst_shape = [src_shape[i] if i in real_axis else 1 for i in range(len(src_shape))]
    else:
        dst_shape = []
        for i in range(len(src_shape)):
            if i not in real_axis:
                dst_shape.append(src_shape[i])
    return dst_shape


@tvm.tag_scope(tag="comm_reduce")
def comm_reduce(data, axis=None, keepdims=False, func=tvm.sum):
    """Reducing the data

    Parameters
    ----------
    data : tvm.Tensor
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
         with size one.
        With this option, the result will broadcast correctly against the input array.

    func : function
        functions like tvm.sum, tvm.max, tvm.min

    Returns
    -------
    ret : tvm.Tensor
    """
    def _build_reduce_compute_func(data, real_axis, reduce_axes, keepdims,
                                   func, *args):
        eval_range = []
        if not keepdims:
            arg_counter = 0
        else:
            arg_counter = None
        red_counter = 0
        for i in range(len(data.shape)):
            if i in real_axis:
                eval_range.append(reduce_axes[red_counter])
                red_counter += 1
            else:
                if not keepdims:
                    eval_range.append(args[arg_counter])
                    arg_counter += 1
                else:
                    eval_range.append(args[i])
        return func(data[tuple(eval_range)], axis=reduce_axes)

    ndim = len(data.shape)
    real_axis = _get_real_axis(ndim, axis)
    if real_axis == list(range(ndim)) and keepdims is False:
        raise ValueError("Currently we do not support all reduce + keepdims = False!"
                         " axis={}, keepdims={}".format(axis, keepdims))
    reduce_axes = [tvm.reduce_axis((0, data.shape[i]), "k%d" %i) for i in real_axis]
    if keepdims:
        target_shape = [tvm.convert(1) if i in real_axis else tvm.convert(data.shape[i])
                        for i in range(ndim)]
    else:
        target_shape = []
        for i in range(ndim):
            if i not in real_axis:
                target_shape.append(tvm.convert(data.shape[i]))
    out = tvm.compute(target_shape,
                      lambda *args: _build_reduce_compute_func(data,
                                                               real_axis,
                                                               reduce_axes,
                                                               keepdims, func, *args),
                      name=data.name + "_red")
    return out


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
    return comm_reduce(data, axis=axis, keepdims=keepdims, func=tvm.sum)


def max(data, axis=None, keepdims=False):
    """Maximum of array elements over a given axis or a list of axes

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
    return comm_reduce(data, axis=axis, keepdims=keepdims, func=tvm.max)


def min(data, axis=None, keepdims=False):
    """Minimum of array elements over a given axis or a list of axes

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
    return comm_reduce(data, axis=axis, keepdims=keepdims, func=tvm.min)
