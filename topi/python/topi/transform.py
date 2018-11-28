# pylint: disable=invalid-name,consider-using-enumerate
"""Injective transformation operators"""
from __future__ import absolute_import as _abs
import tvm
import topi
from . import cpp


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
    return cpp.expand_dims(a, axis, num_newaxis)


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
    return cpp.transpose(a, axes)


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
    return cpp.reshape(a, newshape)


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
    return cpp.squeeze(a, axis)


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
    return cpp.concatenate(a_tuple, axis)


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
    return cpp.split(ary, indices_or_sections, axis)


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


def gather_nd(a, indices):
    """Gather elements from a n-dimension array..

    Parameters
    ----------
    a : tvm.Tensor
        The source array.

    indices : tvm.Tensor
        The indices of the values to extract.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.gather_nd(a, indices)


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


def tensordot(a, b, axes):
    """A generalization of matrix multiplication to tensor.

    Parameters
    ----------
    a : The tensor A
    b : The tensor B
    axes : The number of dimensions to reduce over

    Returns
    -------
    A Tensor computing the result
    """
    if isinstance(axes, int):
        return cpp.tensordot(a, b, axes)
    if isinstance(axes[0], int):
        return cpp.tensordot(a, b, (axes[0],), (axes[1],))
    return cpp.tensordot(a, b, axes[0], axes[1])
