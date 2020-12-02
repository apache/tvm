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
from tvm import te
from tvm import topi
from . import cpp
from . import tag
from .utils import within_index, make_idx, const_vector


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
            "shape inconsistent when expand_like ({}, {}, {})".format(
                len(axis), len(a.shape), len(shape_like.shape)
            )
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


def strided_slice(a, begin, end, strides=None, slice_mode="end"):
    """Slice of an array.

    Parameters
    ----------
    a : tvm.te.Tensor
        The tensor to be sliced.

    begin : list of int
        The indices to begin with in the slicing.

    end : list of int
        Indicies indicating end of the slice.

    strides : list of int, optional
        Specifies the stride values, it can be negative
        in that case, the input tensor will be reversed
        in that particular axis.

    slice_mode : str, optional
        The slice mode [end, size].
        end - The ending indices for the slice [default].
        size - The input strides will be ignored, input end in this mode indicates
        the sizeof a slice starting at the location specified by begin. If end[i]
        is -1, all remaining elements in that dimension are included in the slice.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if (
        isinstance(begin, tvm.te.Tensor)
        or isinstance(end, tvm.te.Tensor)
        or isinstance(strides, tvm.te.Tensor)
    ):
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
    return cpp.strided_slice(a, begin, end, strides, slice_mode)


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
        Indicies indicating end of the slice.

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


def take(a, indices, axis=None, mode="clip"):
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
        return cpp.take(a, indices, mode)
    return cpp.take(a, indices, int(axis), mode)


@tvm.target.generic_func
def take_legalize(attrs, inputs, types):
    """Legalizes dyn.topk op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types
    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    if tvm.relay.ty.is_dynamic(types[0]):
        return tvm.relay.take(tvm.relay.annotation.stop_fusion(inputs[0]), inputs[1], **attrs)
    return None


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


def gather_nd(a, indices):
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


def layout_transform(array, src_layout, dst_layout):
    """Transform the layout according to src_layout and dst_layout

    Parameters
    ----------
    array : tvm.te.Tensor
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

    assert len(data.shape) >= 2, "only support data.ndim >= 2, received data.shape = {}".format(
        data.shape
    )
    assert axis in (0, 1), "only support axis = 0, 1, received axis = {}".format(axis)
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


def unravel_index(indices, shape):
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Example::
    -   unravel_index([22, 41, 37], [7, 6]) = [[3, 6, 6], [4, 5, 1]]

    Parameters
    ----------
    indices : relay.Expr
        An integer array containing indices.

    shape : relay.Expr
        The shape of the array.

    Returns
    -------
    result : relay.Expr
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
    data : relay.Expr
        Input Tensor.

    diagonal : relay.Expr
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
    result : relay.Expr
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
