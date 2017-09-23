"""Injective transformation operators"""
from __future__ import absolute_import as _abs
import tvm
from . import tag

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
