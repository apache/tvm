# pylint: disable=invalid-name,consider-using-enumerate
"""Injective transformation operators"""
from __future__ import absolute_import as _abs
import tvm
import topi
from . import tag
from .util import ravel_index, unravel_index, get_const_int, get_const_tuple
from . import cpp

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


@tvm.tag_scope(tag=tag.BROADCAST)
def expand_like(a, shape_like, axis):
    """Expand an input array with the shape of second array.
    This operation can always be composed of unsqueezing and
    expanding dims on those unsqueezed axes.

    Examples::
    input = [ 12.  19.  27.]
    input.shape = (3,)

    new_shape_array = [[[1,2],[2,3],[1,3]],
                      [[1,4],[4,3],[5,2]],
                      [[7,1],[7,2],[7,3]]]
    new_shape_array.shape = (3, 3, 2)

    expand_like(input, [1,2], new_shape_array) =
                      [[[12,12],[12,12],[12,12]],
                      [[19,19],[19,19],[19,19]],
                      [[27,27],[27,27],[27,27]]]

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be expanded.
    shape_like : tvm.Tensor
        The tensor to with target shape.
    axis: list of int
        axis to be expanded on
    Returns
    -------
    ret : tvm.Tensor
    """
    odim = len(axis) + len(a.shape)
    if odim != len(shape_like.shape):
        if len(a.shape) == 1 and len(axis) == len(shape_like.shape):
            # A special case: `a` is a scalar represented as a 1-dim tensor
            return tvm.compute(shape_like.shape, lambda *idxs: a(0))
        raise ValueError("shape inconsistent when expand_like ({}, {}, {})".format(
            len(axis), len(a.shape), len(shape_like.shape)))

    real_axis = topi.reduction._get_real_axis(len(shape_like.shape), axis)
    real_axis = sorted(real_axis)

    def _compute(*idxs):
        indices = []
        axis_index = 0
        for i in range(0, len(idxs)):
            if i not in real_axis:
                indices.append(idxs[i])
                axis_index += 1
        return a(*indices)
    return tvm.compute(shape_like.shape, _compute)


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

def flip(a, axis=0):
    """Flip/reverse elements of an array in a particular axis.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be expanded.

    axis : int, optional
        The axis along which the tensors will be reveresed.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.flip(a, axis)

def strided_slice(a, begin, end, strides=None):
    """Slice of an array.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be sliced.

    begin: list of int
        The indices to begin with in the slicing.

    end: list of int
        Indicies indicating end of the slice.

    strides: list of int, optional
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.strided_slice(a, begin, end, strides)

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
def squeeze(a, axis=None):
    """Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : tvm.Tensor

    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the shape.
        If an axis is selected with shape entry greater than one, an error is raised.

    Returns
    -------
    squeezed : tvm.Tensor
    """
    a_ndim = len(a.shape)
    a_shape = get_const_tuple(a.shape)
    if axis is None:
        axis = []
        for i, ele in enumerate(a_shape):
            if ele == 1:
                axis.append(i)
    else:
        if isinstance(axis, int):
            axis = axis + a_ndim if axis < 0 else axis
            assert a_shape[axis] == 1
            axis = [axis]
        else:
            axis = [ele + a_ndim if ele < 0 else ele for ele in axis]
            for ele in axis:
                assert a_shape[ele] == 1
    out_shape = []
    search_axis = set(axis)
    for i, a_dim in enumerate(a_shape):
        if i not in search_axis:
            out_shape.append(a_dim)
    if not out_shape:
        out_shape.append(1)
    def _compute(*indices):
        real_indices = []
        flag = 0
        for i in range(a_ndim):
            if i not in search_axis:
                real_indices.append(indices[i - flag])
            else:
                real_indices.append(0)
                flag += 1
        return a(*real_indices)

    return tvm.compute(out_shape, _compute)


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
    assert isinstance(a_tuple, (list, tuple))
    if axis < 0:
        axis += len(a_tuple[0].shape)
    assert axis < len(a_tuple[0].shape)
    axis_sizes = [a_tuple[i].shape[axis] for i in range(len(a_tuple))]
    out_shape = [a_tuple[0].shape[i] for i in range(0, axis)] + [sum(axis_sizes)]\
                + [a_tuple[0].shape[i] for i in range(axis + 1, len(a_tuple[0].shape))]
    out_shape[axis] = tvm.ir_pass.Simplify(out_shape[axis])

    def _compute(*indices):
        ret = a_tuple[0](*indices)
        ind = indices[axis]
        for i in range(len(a_tuple) - 1):
            ind -= axis_sizes[i]
            ret = tvm.select(ind >= 0,
                             a_tuple[i + 1](*(indices[0:axis] + (ind,) + indices[axis + 1:])),
                             ret)
        return ret
    return tvm.compute(out_shape, _compute)


@tvm.tag_scope(tag=tag.INJECTIVE)
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : tvm.Tensor

    indices_or_sections : int or 1-D array

    axis : int

    Returns
    -------
    ret : tuple of tvm.Tensor
    """
    def _compute(begin, *indices):
        real_indices = indices[:axis] + (indices[axis] + begin, ) + indices[axis + 1:]
        return ary(*real_indices)

    if axis < 0:
        axis += len(ary.shape)
    src_axis_size = get_const_int(ary.shape[axis])
    if isinstance(indices_or_sections, int):
        assert indices_or_sections > 0
        assert src_axis_size % indices_or_sections == 0
        seg_size = src_axis_size // indices_or_sections
        begin_ids = [seg_size * i for i in range(indices_or_sections)]
    elif isinstance(indices_or_sections, (tuple, list)):
        assert tuple(indices_or_sections) == tuple(sorted(indices_or_sections)),\
            "Should be sorted, recieved %s" % str(indices_or_sections)
        begin_ids = [0] + list(indices_or_sections)
    else:
        raise NotImplementedError()
    out_shapes = []
    for i in range(len(begin_ids)):
        if i == len(begin_ids) - 1:
            out_axis_size = src_axis_size - begin_ids[i]
        else:
            out_axis_size = begin_ids[i + 1] - begin_ids[i]
        out_shapes.append([ary.shape[i] for i in range(axis)] + [out_axis_size] +\
                          [ary.shape[i] for i in range(axis + 1, len(ary.shape))])
    # pylint: disable=cell-var-from-loop
    return [tvm.compute(out_shape,
                        lambda *indices: _compute(begin_id, *indices), name="s%d" %i)
            for i, (out_shape, begin_id) in enumerate(zip(out_shapes, begin_ids))]
    # pylint: enable=cell-var-from-loop


def take(a, indices, axis=None):
    """Take elements from an array along an axis.

    Parameters
    ----------
    a : tvm.Tensor
        The source array.

    indices : tvm.Tensor
        The indices of the values to extract.

    axis : int, optional
        The axis over which to select values. By default,
        the flattened input array is used.

    Returns
    -------
    ret : tvm.Tensor
    """
    if axis is None:
        return cpp.take(a, indices)
    return cpp.take(a, indices, int(axis))

def matmul(a, b, transp_a=False, transp_b=False):
    """
    Creates an operation that calculates a matrix multiplication (row-major notation):
        A(i, k) * B(k, j)
    if trans_a == trans_b, the usual transposed combinations, otherwise

    Parameters
    ----------
    a : The matrix A
    b : The matrix B
    trans_a : Is A's layout transposed?
    trans_b : Is B's layout transposed?

    Returns
    -------
    A Tensor whose op member is the matmul operation
    """
    return cpp.matmul(a, b, transp_a, transp_b)
