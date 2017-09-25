# pylint: disable=invalid-name
"""Injective transformation operators"""
from __future__ import absolute_import as _abs
import tvm
from . import tag
from .util import ravel_index, unravel_index, get_const_int

@tvm.tag_scope(tag=tag.BROADCAST)
def expand_dims(a, axis, num_newaxis=1):
    """Expand the shape of an array.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be expanded.

    num_newaxis: int, optional
        Number of newaxis to be inserted on axis

    Returns
    -------
    ret : tvm.Tensor
    """
    axis = len(a.shape) + axis + 1 if axis < 0 else axis
    new_shape = a.shape[:axis] + ([1] * num_newaxis) + a.shape[axis:]
    def _compute(*indices):
        idx = indices[:axis] + indices[axis + num_newaxis:]
        return a(*idx)
    return tvm.compute(new_shape, _compute)


@tvm.tag_scope(tag=tag.INJECTIVE)
def transpose(a, axes=None):
    """Permute the dimensions of an array.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be expanded.

    axes: tuple of ints, optional
        By default, reverse the dimensions.

    Returns
    -------
    ret : tvm.Tensor
    """
    ndim = len(a.shape)
    axes = axes if axes else tuple(reversed(range(ndim)))
    new_shape = [a.shape[x] for x in axes]
    def _compute(*indices):
        idx = [1] * len(axes)
        for i, k in enumerate(axes):
            idx[k] = indices[i]
        return a(*idx)
    return tvm.compute(new_shape, _compute)


@tvm.tag_scope(tag=tag.INJECTIVE)
def reshape(a, newshape):
    """Reshape the array

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be reshaped
    newshape : tuple of ints
        The new shape

    Returns
    -------
    ret : tvm.Tensor
    """
    ndim = len(a.shape)
    a_shape = [a.shape[i] for i in range(ndim)]
    return tvm.compute(newshape,
                       lambda *indices: a(*unravel_index(ravel_index(indices, newshape), a_shape)))


@tvm.tag_scope(tag=tag.INJECTIVE)
def concatenate(a_tuple, axis=0):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a_tuple : tuple of tvm.Tensor
        The arrays to concatenate
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    ret : tvm.Tensor
    """
    if axis < 0:
        axis += len(a_tuple[0].shape)
    assert axis < len(a_tuple[0].shape)
    axis_sizes = [a_tuple[i].shape[axis] for i in range(len(a_tuple))]
    out_shape = [a_tuple[0].shape[i] for i in range(0, axis)] + [sum(axis_sizes)]\
                + [a_tuple[0].shape[i] for i in range(axis + 1, len(a_tuple[0].shape))]

    def _compute(*indices):
        ret = a_tuple[0](*indices)
        ind = indices[axis]
        for i in range(len(a_tuple) - 1):
            ind -= axis_sizes[i]
            ret = tvm.select(ind >= axis_sizes[i],
                             a_tuple[i + 1](*(indices[0:axis] + (ind,) + indices[axis + 1:])),
                             ret)
        return ret
    return tvm.compute(out_shape, _compute)
