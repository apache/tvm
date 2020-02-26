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
# pylint: disable=invalid-name,consider-using-enumerate
"""Injective transformation operators"""
from __future__ import absolute_import as _abs
import tvm
import topi
from . import cpp
from . import tag
from .util import within_index, make_idx


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
    if strides is None:
        strides = []
    return cpp.strided_slice(a, begin, end, strides)

@tvm.tag_scope(tag=tag.INJECTIVE+",strided_set")
def strided_set(a, v, begin, end, strides=None):
    """Set slice of an array.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be sliced.

    v : tvm.Tensor
        The values to set

    begin: tvm.Tensor
        The indices to begin with in the slicing.

    end: tvm.Tensor
        Indicies indicating end of the slice.

    strides: tvm.Tensor, optional
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    Returns
    -------
    ret : tvm.Tensor
    """
    n = len(a.shape)

    if len(begin.shape) != 1:
        raise ValueError("begin should be a vector")
    if not begin.dtype == 'int32':
        raise TypeError("begin should be int32")
    if len(end.shape) != 1:
        raise ValueError("end should be a vector")
    if not end.dtype == 'int32':
        raise TypeError("end should be int32")
    if strides is not None:
        if len(strides.shape) != 1:
            raise ValueError("strides should be a vector")
        if not strides.dtype == 'int32':
            raise TypeError("strides should be int32")

    def _max(a, b):
        return tvm.expr.Select(a > b, a, b)

    if strides is None:
        strides = [tvm.const(1, 'int32')] * n
    else:
        strides = [tvm.if_then_else(strides.shape[0] > i,
                                    strides[i],
                                    tvm.const(1, 'int32'))
                   for i in range(n)]

    begin = [tvm.if_then_else(begin.shape[0] > i,
                              begin[i],
                              tvm.expr.Select(strides[i] > 0,
                                              tvm.const(0, 'int32'),
                                              a.shape[i]))
             for i in range(n)]
    end = [tvm.if_then_else(end.shape[0] > i,
                            end[i],
                            tvm.expr.Select(strides[i] > 0,
                                            a.shape[i] + 1,
                                            -(a.shape[i] + 1)))
           for i in range(n)]


    # Convert negative indexes
    for i in range(n):
        begin[i] = tvm.if_then_else(begin[i] < 0,
                                    begin[i] + a.shape[i],
                                    begin[i])
        end[i] = tvm.if_then_else(end[i] < 0,
                                  end[i] + a.shape[i],
                                  end[i])

    def _select(*indices):
        from_val = []
        index_tuple = []
        for i in range(n):
            from_val.append(within_index(begin[i], end[i], strides[i], indices[i]))
            index_tuple.append(
                make_idx(begin[i], end[i], strides[i], a.shape[i], indices[i]))
        return tvm.if_then_else(tvm.all(*from_val), v(*index_tuple), a(*indices))

    return tvm.compute(a.shape, _select, name="strided_set")


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


def stack(a, axis):
    """Repeats the whole array multiple times.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be stacked.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.


    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.stack(a, axis)


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


def take(a, indices, axis=None, mode="clip"):
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

    mode : str, optional
        Specifies how out-of-bound indices will behave.
        clip - clip to the range (default)
        wrap - wrap around the indices
        fast - no clip or wrap around (user must make sure indices are in-bound)

    Returns
    -------
    ret : tvm.Tensor
    """
    if axis is None:
        return cpp.take(a, indices, mode)
    return cpp.take(a, indices, int(axis), mode)


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


def arange(start, stop=None, step=1, dtype="float32"):
    """Creates a tensor with evenly spaced values within a given interval.

    Parameters
    ----------
    start : tvm.Expr, optional
        Start of interval. The interval includes this value. The default start
        value is 0.

    stop : tvm.Expr
        Stop of interval. The interval does not include this value.

    step : tvm.Expr, optional
        Spacing between values. The default step size is 1.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.Tensor
        The resulting tensor.
    """
    if stop is None:
        stop = start
        start = 0
    return cpp.arange(start, stop, step, dtype)


def repeat(a, repeats, axis):
    """Repeats elements of an array.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be repeated.

    repeats: int, required
        Number of repetitions for each element

    axis: int, optional
        The axis along which to repeat values

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.repeat(a, repeats, axis)


def tile(a, reps):
    """Repeats the whole array multiple times.

    Parameters
    ----------
    a : tvm.Tensor
        The tensor to be tiled.

    reps: tuple of ints, required
        The number of times for repeating the tensor

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.tile(a, reps)


def layout_transform(array, src_layout, dst_layout):
    """Transform the layout according to src_layout and dst_layout

    Parameters
    ----------
    array : tvm.Tensor
        The source array.

    src_layout : str
        the source layout.

    dst_layout : str
        the destination layout.
    """
    return cpp.layout_transform(array, src_layout, dst_layout)


def shape(array, dtype="int32"):
    """Get the shape of input array

    Parameters
    ----------
    array : tvm.Tensor
        The source tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.Tensor
        The resulting tensor.
    """
    return cpp.shape(array, dtype)


def sequence_mask(data, valid_length, mask_value=0, axis=0):
    """Sets all elements outside the expected length of the sequence to a constant value.

    This function takes an n-dimensional input array of the form [MAX_LENGTH, batch_size, ...] or
    [batch_size, MAX_LENGTH, ...] and returns an array of the same shape.

    `axis` means the axis of the length dimension and can only be 0 or 1. If `axis` is 0,
    the data must have shape [MAX_LENGTH, batch_size, ...]. Otherwise (axis=1), the data must have
    shape [batch_size, MAX_LENGTH, ...].

    `valid_length` gives the length of each sequence. `valid_length` should be
    a 1D int array with positive ints and has dimension [batch_size,].

    Parameters
    ----------
    data : tvm.Tensor
        N-D with shape [MAX_LENGTH, batch_size, ...] or [batch_size, MAX_LENGTH, ...]
        depending on the value of `axis`.

    valid_length : tvm.Tensor
        1-D with shape [batch_size,]

    mask_value : float, optional
        The masking value, default 0

    axis : int, optional
        axis of the length dimension, must be 0 or 1, default 0

    Returns
    -------
    output : tvm.Tensor
        N-D with shape [MAX_LENGTH, batch_size, ...] or [batch_size, MAX_LENGTH, ...]
        depending on the value of `axis`.
    """

    assert len(data.shape) >= 2,\
        "only support data.ndim >= 2, received data.shape = {}".format(data.shape)
    assert axis in (0, 1), "only support axis = 0, 1, received axis = {}".format(axis)
    return cpp.sequence_mask(data, valid_length, mask_value, axis)


def ndarray_size(array, dtype="int32"):
    """Get the number of elements of input array

    Parameters
    ----------
    array : tvm.Tensor
        The source tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.Tensor
        The resulting tensor.
    """
    return cpp.ndarray_size(array, dtype)


def where(condition, x, y):
    """Get the elements, either from x or y, depending on the condition.

    Parameters
    ----------
    condition : tvm.Tensor
        The condition array.

    x : tvm.Tensor
        First array to be selected.

    y : tvm.Tensor
        Second array to be selected.

    Returns
    -------
    result : tvm.Tensor
        A Tensor selected from x or y depending on condition.
    """
    return cpp.where(condition, x, y)

def one_hot(indices, on_value, off_value, depth, axis, dtype):
    """
    Returns a one-hot tensor where the locations repsented by indices take value on_value,
    other locations take value off_value.
    Final dimension is <indices outer dimensions> x depth x <indices inner dimensions>.

    Parameters
    ----------
    indices : tvm.Tensor
        Locations to set to on_value.

    on_value : tvm.Tensor
        Value to fill at indices.

    off_value : tvm.Tensor
        Value to fill at all other positions besides indices.

    depth : int
        Depth of the one-hot dimension.

    axis : int
        Axis to fill.

    dtype : relay.DataType
        Data type of the output tensor.

    Returns
    -------
    ret : relay.Expr
        The one-hot tensor.

    Examples
    --------
    .. code-block:: python

        indices = [0, 1, 2]

        relay.one_hot(indices, 3) =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    return cpp.one_hot(indices, on_value, off_value, depth, axis, dtype)
