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
# pylint: disable=invalid-name,consider-using-enumerate,redefined-outer-name
"""Injective transformation operators"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te, topi

from . import cpp, tag
from .utils import const_vector, make_idx, within_index


def expand_dims(a, axis, num_newaxis=1):
    """Expand the shape of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be expanded.

    num_newaxis: int, optional
        Number of newaxis to be inserted on axis

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.expand_dims(a, axis, num_newaxis)


def expand_like(a, shape_like, axis):
    """Expand an input array with the shape of second array.
    This operation can always be composed of unsqueezing and
    expanding dims on those unsqueezed axes.

    Examples
    --------
    .. code-block::

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
    a : tvm.te.Tensor
        The tensor to be expanded.
    shape_like : tvm.te.Tensor
        The tensor to with target shape.
    axis: list of int
        axis to be expanded on

    Returns
    -------
    ret : tvm.te.Tensor
    """
    odim = len(axis) + len(a.shape)
    if odim != len(shape_like.shape):
        if len(a.shape) == 1 and len(axis) == len(shape_like.shape):
            # A special case: `a` is a scalar represented as a 1-dim tensor
            return te.compute(shape_like.shape, lambda *idxs: a(0))
        raise ValueError(
            f"shape inconsistent when expand_like ({len(axis)}, "
            f"{len(a.shape)}, {len(shape_like.shape)})"
        )

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

    return te.compute(shape_like.shape, _compute)


def transpose(a, axes=None):
    """Permute the dimensions of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be expanded.

    axes: tuple of ints, optional
        By default, reverse the dimensions.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.transpose(a, axes)


def flip(a, axis=0):
    """Flip/reverse elements of an array in a particular axis.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be expanded.

    axis : int, optional
        The axis along which the tensors will be reveresed.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.flip(a, axis)


def reverse_sequence(a, seq_lengths, seq_axis=1, batch_axis=0):
    """Reverse the tensor for variable length slices.
    Input is first sliced along batch axis and then elements are reversed along seq axis.

    Parameters
    ----------
    a : tvm.te.Tensor
       The tensor to be reversed.

    seq_lengths : tvm.te.Tensor
       A 1D Tensor with length a.dims[batch_axis]
       Must be one of the following types: int32, int64
       if seq_lengths[i] > a.dims[seq_axis], it is rounded to a.dims[seq_axis]
       if seq_lengths[i] < 1, it is rounded to 1

    seq_axis : int, optional
       The axis along which the elements will be reversed. Default is 1.

    batch_axis : int, optional
       The axis along which the tensor will be sliced. Default is 0.

    Returns
    -------
    ret : tvm.te.Tensor
       The computed result of same shape and type as of input.

    """
    return cpp.reverse_sequence(a, seq_lengths, seq_axis, batch_axis)


def strided_slice(a, begin, end, strides=None, axes=None, slice_mode="end", assume_inbound=True):
    """Slice of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be sliced.

    begin : list of int
        The indices to begin with in the slicing.

    end : list of int
        Indices indicating end of the slice.

    strides : list of int, optional
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    axes : list of int, optional
        Axes along which slicing is applied. When it is specified, begin, end
        strides, and axes need to a list of integers of the same length.

    slice_mode : str, optional
        The slice mode [end, size].
        end - The ending indices for the slice [default].
        size - The input strides will be ignored, input end in this mode indicates
        the sizeof a slice starting at the location specified by begin. If end[i]
        is -1, all remaining elements in that dimension are included in the slice.

    assume_inbound: bool, optional
        A flag to indicate if all indices are assumed to be inbound

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if (
        isinstance(begin, tvm.te.Tensor)
        or isinstance(end, tvm.te.Tensor)
        or isinstance(strides, tvm.te.Tensor)
    ):
        assert axes is None, "axes argument is not supported by dynamic strided slice yet."
        if not isinstance(begin, tvm.te.Tensor):
            begin = const_vector(begin)
        if not isinstance(end, tvm.te.Tensor):
            end = const_vector(end)
        if strides is None:
            strides = [1] * begin.shape[0].value
        if not isinstance(strides, tvm.te.Tensor):
            strides = const_vector(strides)
        return cpp.dynamic_strided_slice(a, begin, end, strides)
    if strides is None:
        strides = []
    if axes is None:
        axes = []
    return cpp.strided_slice(a, begin, end, strides, axes, slice_mode, assume_inbound)


def dynamic_strided_slice(a, begin, end, strides, output_shape):
    """Slice of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be sliced.

    begin : tvm.te.Tensor
        The indices to begin with in the slicing.

    end : tvm.te.Tensor
        Indices indicating end of the slice.

    strides : tvm.te.Tensor
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    output_shape: list of PrimExpr
        Specifies the output shape

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(begin, tvm.te.Tensor):
        begin = const_vector(begin)
    if not isinstance(end, tvm.te.Tensor):
        end = const_vector(end)
    if not isinstance(strides, tvm.te.Tensor):
        strides = const_vector(strides)
    return cpp.relax_dynamic_strided_slice(a, begin, end, strides, output_shape)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",strided_set")
def strided_set(a, v, begin, end, strides=None):
    """Set slice of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be sliced.

    v : tvm.te.Tensor
        The values to set

    begin: tvm.te.Tensor
        The indices to begin with in the slicing.

    end: tvm.te.Tensor
        Indices indicating end of the slice.

    strides: tvm.te.Tensor, optional
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    n = len(a.shape)

    if len(begin.shape) != 1:
        raise ValueError("begin should be a vector")
    if not begin.dtype == "int32":
        raise TypeError("begin should be int32")
    if len(end.shape) != 1:
        raise ValueError("end should be a vector")
    if not end.dtype == "int32":
        raise TypeError("end should be int32")
    if strides is not None:
        if len(strides.shape) != 1:
            raise ValueError("strides should be a vector")
        if not strides.dtype == "int32":
            raise TypeError("strides should be int32")

    def _max(a, b):
        return tvm.tir.Select(a > b, a, b)

    if strides is None:
        strides = [tvm.tir.const(1, "int32")] * n
    else:
        strides = [
            tvm.tir.if_then_else(strides.shape[0] > i, strides[i], tvm.tir.const(1, "int32"))
            for i in range(n)
        ]

    begin = [
        tvm.tir.if_then_else(
            begin.shape[0] > i,
            begin[i],
            tvm.tir.Select(strides[i] > 0, tvm.tir.const(0, "int32"), a.shape[i]),
        )
        for i in range(n)
    ]
    end = [
        tvm.tir.if_then_else(
            end.shape[0] > i,
            end[i],
            tvm.tir.Select(strides[i] > 0, a.shape[i] + 1, -(a.shape[i] + 1)),
        )
        for i in range(n)
    ]

    # Convert negative indexes
    for i in range(n):
        begin[i] = tvm.tir.if_then_else(begin[i] < 0, begin[i] + a.shape[i], begin[i])
        end[i] = tvm.tir.if_then_else(end[i] < 0, end[i] + a.shape[i], end[i])

    def _select(*indices):
        from_val = []
        index_tuple = []
        for i in range(n):
            from_val.append(within_index(begin[i], end[i], strides[i], indices[i]))
            index_tuple.append(make_idx(begin[i], end[i], strides[i], a.shape[i], indices[i]))
        return tvm.tir.if_then_else(tvm.tir.all(*from_val), v(*index_tuple), a(*indices))

    return te.compute(a.shape, _select, name="strided_set")


def reshape(a, newshape):
    """Reshape the array

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be reshaped
    newshape : tuple of ints
        The new shape

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.reshape(a, newshape)


def squeeze(a, axis=None):
    """Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : tvm.te.Tensor

    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the shape.
        If an axis is selected with shape entry greater than one, an error is raised.

    Returns
    -------
    squeezed : tvm.te.Tensor
    """
    return cpp.squeeze(a, axis)


def concatenate(a_tuple, axis=0):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a_tuple : tuple of tvm.te.Tensor
        The arrays to concatenate

    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.concatenate(a_tuple, axis)


def stack(a, axis):
    """Repeats the whole array multiple times.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be stacked.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.


    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.stack(a, axis)


def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : tvm.te.Tensor

    indices_or_sections : int or 1-D array

    axis : int

    Returns
    -------
    ret : tuple of tvm.te.Tensor
    """
    return cpp.split(ary, indices_or_sections, axis)


def take(a, indices, axis=None, batch_dims=0, mode="clip"):
    """Take elements from an array along an axis.

    Parameters
    ----------
    a : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    axis : int, optional
        The axis over which to select values. By default,
        the flattened input array is used.

    batch_dims : int
        The number of batch dimensions. By default is 0.

    mode : str, optional
        Specifies how out-of-bound indices will behave.
        clip - clip to the range (default)
        wrap - wrap around the indices
        fast - no clip or wrap around (user must make sure indices are in-bound)

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if axis is None:
        return cpp.take(a, indices, int(batch_dims), mode)
    return cpp.take(a, indices, int(batch_dims), int(axis), mode)


def gather(data, axis, indices):
    """Gather values along given axis from given indices.

    E.g. for a 3D tensor, output is computed as:

    .. code-block:: python

        out[i][j][k] = data[indices[i][j][k]][j][k]  # if axis == 0
        out[i][j][k] = data[i][indices[i][j][k]][k]  # if axis == 1
        out[i][j][k] = data[i][j][indices[i][j][k]]  # if axis == 2

    ``indices`` must have same shape as ``data``, except at dimension ``axis``
    which must just be not null. Output will have same shape as ``indices``.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis: int
        The axis along which to index.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.gather(data, axis, indices)


def gather_nd(a, indices, batch_dims=0):
    """Gather elements from a n-dimension array..

    Parameters
    ----------
    a : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.gather_nd(a, indices, batch_dims)


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
    result : tvm.te.Tensor
        The resulting tensor.
    """
    if stop is None:
        stop = start
        start = 0
    return cpp.arange(start, stop, step, dtype)


def meshgrid(a_tuple, indexing):
    """Create coordinate matrices from coordinate vectors.

    Parameters
    ----------
    a_tuple : tuple of tvm.te.Tensor
        The coordinate vectors or scalars.

    indexing : str
        Indexing mode, either "ij" or "xy".

    Returns
    -------
    result : tuple of tvm.te.Tensor
        The resulting grids for each axis.
    """
    return cpp.meshgrid(a_tuple, indexing)


def repeat(a, repeats, axis):
    """Repeats elements of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be repeated.

    repeats: int, required
        Number of repetitions for each element

    axis: int, optional
        The axis along which to repeat values

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.repeat(a, repeats, axis)


def tile(a, reps):
    """Repeats the whole array multiple times.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be tiled.

    reps: tuple of ints, required
        The number of times for repeating the tensor

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return cpp.tile(a, reps)


def layout_transform(array, src_layout, dst_layout, schedule_rule="None"):
    """Transform the layout according to src_layout and dst_layout

    Parameters
    ----------
    array : tvm.te.Tensor
        The source array.

    src_layout : str
        the source layout.

    dst_layout : str
        the destination layout.

    schedule_rule : str
        the schedule rule to apply if any
    """
    return cpp.layout_transform(array, src_layout, dst_layout, schedule_rule)


def shape(array, dtype="int32"):
    """Get the shape of input array

    Parameters
    ----------
    array : tvm.te.Tensor
        The source tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.te.Tensor
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
    data : tvm.te.Tensor
        N-D with shape [MAX_LENGTH, batch_size, ...] or [batch_size, MAX_LENGTH, ...]
        depending on the value of `axis`.

    valid_length : tvm.te.Tensor
        1-D with shape [batch_size,]

    mask_value : float, optional
        The masking value, default 0

    axis : int, optional
        axis of the length dimension, must be 0 or 1, default 0

    Returns
    -------
    output : tvm.te.Tensor
        N-D with shape [MAX_LENGTH, batch_size, ...] or [batch_size, MAX_LENGTH, ...]
        depending on the value of `axis`.
    """

    assert len(data.shape) >= 2, f"only support data.ndim >= 2, received data.shape = {data.shape}"
    assert axis in (0, 1), f"only support axis = 0, 1, received axis = {axis}"
    return cpp.sequence_mask(data, valid_length, mask_value, axis)


def ndarray_size(array, dtype="int32"):
    """Get the number of elements of input array

    Parameters
    ----------
    array : tvm.te.Tensor
        The source tensor.

    dtype : str, optional
        The target data type.

    Returns
    -------
    result : tvm.te.Tensor
        The resulting tensor.
    """
    return cpp.ndarray_size(array, dtype)


def where(condition, x, y):
    """Get the elements, either from x or y, depending on the condition.

    Parameters
    ----------
    condition : tvm.te.Tensor
        The condition array.

    x : tvm.te.Tensor
        First array to be selected.

    y : tvm.te.Tensor
        Second array to be selected.

    Returns
    -------
    result : tvm.te.Tensor
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
    indices : tvm.te.Tensor
        Locations to set to on_value.

    on_value : tvm.te.Tensor
        Value to fill at indices.

    off_value : tvm.te.Tensor
        Value to fill at all other positions besides indices.

    depth : int
        Depth of the one-hot dimension.

    axis : int
        Axis to fill.

    dtype : str
        Data type of the output tensor.

    Returns
    -------
    ret : tvm.te.Tensor
        The one-hot tensor.

    Examples
    --------
    .. code-block:: python

        indices = [0, 1, 2]

        topi.one_hot(indices, 3) =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    return cpp.one_hot(indices, on_value, off_value, depth, axis, dtype)


def unravel_index(indices, shape):
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Example::
    -   unravel_index([22, 41, 37], [7, 6]) = [[3, 6, 6], [4, 5, 1]]

    Parameters
    ----------
    indices : tvm.te.Tensor
        An integer array containing indices.

    shape : tvm.te.Tensor
        The shape of the array.

    Returns
    -------
    result : tvm.te.Tensor
        The tuple of coordinate arrays.
    """

    return cpp.unravel_index(indices, shape)


def sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value=0):
    """Converts a sparse representation into a dense tensor.

    Example::
    -   sparse_to_dense([[0, 0], [1, 1]], [2, 2], [3, 3], 0) = [[3, 0], [0, 3]]

    Parameters
    ----------
    sparse_indices : tvm.te.Tensor
        A 0-D, 1-D, or 2-D tensor of integers containing location of sparse values.

    output_shape : A list of integers
        Shape of the dense output tensor.

    sparse_values : tvm.te.Tensor
        A 0-D or 1-D tensor containing the sparse values for the sparse indices.

    default_value : tvm.te.Tensor
        A 0-D tensor containing the default value for the remaining locations.
        Defaults to 0.

    Returns
    -------
    result : tvm.te.Tensor
        Dense tensor of shape output_shape. Has the same type as sparse_values.
    """

    return cpp.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)


def matrix_set_diag(data, diagonal, k=0, align="RIGHT_LEFT"):
    """
    Returns a tensor with the diagonals of input tensor replaced with the provided diagonal values.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input Tensor.

    diagonal : tvm.te.Tensor
        Values to be filled in the diagonal.

    k : int or tuple of int, optional
        Diagonal Offset(s). The diagonal or range of diagonals to set. (0 by default)
        Positive value means superdiagonal, 0 refers to the main diagonal, and
        negative value means subdiagonals. k can be a single integer (for a single diagonal)
        or a pair of integers specifying the low and high ends of a matrix band.
        k[0] must not be larger than k[1].

    align : string, optional
        Some diagonals are shorter than max_diag_len and need to be padded.
        align is a string specifying how superdiagonals and subdiagonals should be aligned,
        respectively. There are four possible alignments: "RIGHT_LEFT" (default), "LEFT_RIGHT",
        "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals to the right
        (left-pads the row) and subdiagonals to the left (right-pads the row). It is the packing
        format LAPACK uses. cuSPARSE uses "LEFT_RIGHT", which is the opposite alignment.

    Returns
    -------
    result : tvm.te.Tensor
        New tensor with given diagonal values.

    Examples
    --------
    .. code-block:: python

        data = [[[7, 7, 7, 7],
                 [7, 7, 7, 7],
                 [7, 7, 7, 7]],
                [[7, 7, 7, 7],
                 [7, 7, 7, 7],
                 [7, 7, 7, 7]]]

        diagonal = [[1, 2, 3],
                    [4, 5, 6]]

        topi.matrix_set_diag(input, diagonal) =
            [[[1, 7, 7, 7],
              [7, 2, 7, 7],
              [7, 7, 3, 7]],
             [[4, 7, 7, 7],
              [7, 5, 7, 7],
              [7, 7, 6, 7]]]
    """
    if isinstance(k, (tuple, list)):
        k_one = k[0]
        if len(k) >= 2:
            k_two = k[1]
        else:
            k_two = k[0]
    else:
        k_one = k
        k_two = k

    super_diag_right_align = align[:5] == "RIGHT"
    sub_diag_right_align = align[-5:] == "RIGHT"

    return cpp.matrix_set_diag(
        data, diagonal, k_one, k_two, super_diag_right_align, sub_diag_right_align
    )


def adv_index(data, indices):
    """Numpy style indexing with tensors.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data.

    indices : A list of tvm.te.Tensor
        Tensor index.

    Returns
    -------
    result : tvm.te.Tensor
        Output tensor
    """
    return cpp.adv_index(data, indices)


def sliding_window(data, axis, window_shape, strides):
    """Slide a window over the data tensor.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis : int
        What axis the window begins sliding over. Window will be slid over
        this axis and all following axes. The axis value determines the window
        shape (and thus, the number of strides): window shape and strides must
        both be of length `data.ndim-axis`.

    window_shape : List[int]
        The window shape to form over the input. Window shape must be of length
        `data.ndim-axis`.

    strides : List[int]
        How to stride the window along each dimension. Strides must be of length
        `data.ndim-axis`.

    Returns
    -------
    result : tvm.te.Tensor
        The resulting tensor.
    """
    return cpp.sliding_window(data, axis, window_shape, strides)


def trilu(data, k, upper):
    """
    Given a 2-D matrix or batches of 2-D matrices, returns the
    upper or lower triangular part of the tensor.

    Parameters
    ----------
    data: tvm.te.Tensor
        The tensor that trilu will be applied to. Must be either
        a 2D matrix or a tensor of batches of 2D matrices.

    k: tvm.te.Tensor
        The number of diagonals above or below the main diagonal
        to exclude or include.

    upper: bool
        If True, only upper triangular values of input are kept,
        if False, the lower triangular values are kept.


    Returns
    -------
    ret : tvm.te.Tensor
        The new tensor with appropriate diagonals set to zero.

    Examples
    --------
    .. code-block:: python

        x = [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]]

        topi.trilu(x, True, 0) =
            [[0, 1, 2],
             [0, 4, 5],
             [0, 0, 8]]
    """
    # Make sure datatype is consistent.
    if k.dtype != "int32":
        k = tvm.tir.Cast("int32", k)

    # Check either above or below diagonal depending on upper.
    check_op = tvm.tir.GE
    if upper:
        check_op = tvm.tir.LE

    def _apply_trilu(*indices):
        row_index = indices[-2]
        col_index = indices[-1]
        # promote row & col indices
        if row_index.dtype != col_index.dtype:
            target_type = (col_index + row_index).dtype
            if row_index.dtype != target_type:
                row_index = tvm.tir.Cast(target_type, row_index)
            else:
                col_index = tvm.tir.Cast(target_type, col_index)
        other_indices = indices[:-2]
        check_position = check_op(row_index, col_index - k)
        value = data(*other_indices, row_index, col_index)
        return tvm.tir.Select(check_position, value, tvm.tir.const(0, data.dtype))

    return te.compute(data.shape, _apply_trilu, name="trilu", tag=topi.tag.ELEMWISE)

def index_tensor(data, indices):
    """ TODO docstring  
    - If 'indices' is a list/tuple of length > 1, we interpret that as multiple advanced indices,
      and implement with topi.adv_index (plus negative-index correction if desired).
    - Otherwise, interpret 'indices' as a single advanced index, and implement with topi.take.

    Replicate data[indices] using only:
    - basic indexing on data
    - torch.index_select
    - concatenation/stack
    - broadcasting
    … and no advanced indexing.

    Approach for multiple advanced indices: broadcast and loop

    Approach for single advanced index: 
    1. Convert the nested Python list to a LongTensor.
    2. Remove exactly one leading dimension of size=1, if present. (Matches PyTorch's shape rule.)
    3. Flatten -> fix negative indices -> index_select -> reshape.
    """
    # The typical pattern is to define the new output via te.compute,
    # with a lambda that describes the element-wise operation.
    # return te.compute(
    #     data.shape,
    #     lambda *indices: data(*indices) + tvm.tir.const(1, data.dtype),
    #     name="dummy_add_one",
    #     # For a simple element-wise operator, you can use tag=topi.tag.ELEMWISE
    #     tag="elemwise",
    # ) # TODO this also works

    # return topi.sum(data, axis=[0]) # TODO this also works

    # return data

    # TODO uncomment


    # Helper to fix negative indices:  out_idx = where(idx<0, idx+dim_size, idx)
    def _fix_negatives(idx_t, dim_size):
        #  idx_t, dim_size are tvm.te.Tensor or integers.
        #  We'll broadcast if needed.  We can do so by calling topi.where(...) with the condition
        #   (idx_t < 0).
        #  For static shape, `dim_size` could be int.  For dynamic shape, dim_size might be a Tensor.
        #  Suppose dim_size is int here. Then we can just do:

        # TODO uncomment
        zero_t = topi.full_like(idx_t, 0)
        dim_size_t = topi.full_like(idx_t, dim_size)  # broadcast if needed
        return topi.where(topi.less(idx_t, zero_t), topi.add(idx_t, dim_size_t), idx_t)

    # --- Check whether indices is multiple advanced indices or single advanced index. ---
    if isinstance(indices, (list, tuple)) and len(indices) > 1:
        # -----------------------------------------------------------
        # CASE B: multiple advanced indices
        # -----------------------------------------------------------
        # Suppose each sub_i is a tvm.te.Tensor of integer type, indexing a separate dimension.
        # We want to broadcast them to a common shape (if not already),
        # fix negative indices, then use topi.adv_index.
        idx_list = list(indices)

        # 1) Determine broadcast shape. For simplicity we can rely on `topi.adv_index` automatically
        #    broadcasting the indices if they differ in shape. If you need explicit broadcasting,
        #    you can do so via topi utilities (e.g. topi.broadcast_to).
        #    Then fix negative indexing dimensionwise.
        #    data.shape is e.g. [d0, d1, d2, ...], so for the i-th advanced index, dimension = data.shape[i].
        #    We fix negative indexing if desired:
        final_indices = []
        for i, idx_t in enumerate(idx_list):
            # If we want negative fix, do it here:
            dim_size = data.shape[i]  # a PrimExpr
            fixed = _fix_negatives(idx_t, dim_size)
            final_indices.append(fixed)

        # 2) Use topi.adv_index
        #    This will produce a new tensor with shape = broadcast of final_indices.
        result = topi.adv_index(data, final_indices)
        return result

    else:
        # -----------------------------------------------------------
        # CASE A: single advanced index
        # -----------------------------------------------------------
        # We interpret 'indices' as a single integer-tensor for dimension=0 indexing.
        # So the result shape is [*indices_shape, leftover_dims], with leftover_dims = data.shape[1:].
        #
        # Steps, paralleling the Python:
        #  1) If the first dimension of indices is 1, remove it   => topi.squeeze if we want.
        #  2) Flatten => topi.reshape
        #  3) fix negative indices => topi.where
        #  4) gather => topi.take(..., axis=0)
        #  5) reshape => combine advanced dims + leftover dims
        idx_t = indices if isinstance(indices, te.Tensor) else indices[0]
        return topi.sum(data, axis=[0]) # TODO this also works
    

        # Possibly remove leading dimension if shape[0]==1:
        if len(idx_t.shape) > 0:
            first_dim = idx_t.shape[0]
            if isinstance(first_dim, int) and first_dim == 1:
                # topi.squeeze can remove exactly one axis:
                idx_t = topi.squeeze(idx_t, axis=[0])
            else:
                # If we suspect it's dynamic, we can check with a small schedule or approach,
                # but here's the naive approach: we skip if the dimension is unknown
                pass

        # Flatten
        flattened = topi.reshape(idx_t, (-1,))

        # fix negative indexing
        # data.shape[0] is batch dimension
        fixed = _fix_negatives(flattened, data.shape[0])

        # gather => topi.take
        # out shape = [len_of_fixed] + leftover_dims
        picked = topi.take(data, fixed, axis=0)

        # final reshape => idx_t original shape (after squeeze) + leftover
        # we can get idx_t's shape with topi.shape if dynamic, or known statically
        adv_shape = tuple(idx_t.shape)  # or topi.shape(idx_t) if dynamic
        leftover_dims = tuple(data.shape[1:])
        print('adv_shape type', type(adv_shape))
        print('leftover_dims type', type(leftover_dims))
        print("A #############################################")
        final_shape = adv_shape + leftover_dims
        print("B #############################################")
        result = topi.reshape(picked, final_shape)
        print("C #############################################")
        return result
