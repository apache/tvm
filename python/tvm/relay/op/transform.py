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

# pylint: disable=import-outside-toplevel
"""Transform operators."""

from . import _make
from .dyn import _make as _dyn_make
from .tensor import shape_of
from ..expr import TupleWrapper, const, Expr, Tuple
from ...tir import expr as _expr


def cast(data, dtype):
    """Cast input tensor to data type.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    dtype: str
        The target data type

    Returns
    -------
    result : relay.Expr
        The casted result.
    """
    from .. import _ffi_api as _relay_make

    return _relay_make.cast(data, dtype)


def cast_like(data, dtype_like):
    """Cast input tensor to data type of another tensor.
    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.
    dtype_like: relay.Expr
        The tensor to cast to.
    Returns
    -------
    result : relay.Expr
        The casted result.
    """
    from .. import _ffi_api as _relay_make

    return _relay_make.cast_like(data, dtype_like)


def reinterpret(data, dtype):
    """Reinterpret input tensor to data type.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    dtype: str
        The target data type

    Returns
    -------
    result : relay.Expr
        The reinterpreted result.
    """
    from .. import _make as _relay_make

    return _relay_make.reinterpret(data, dtype)


def expand_dims(data, axis, num_newaxis=1):
    """Insert `num_newaxis` axises at the position given by `axis`.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : int
        The axis at which the input array is expanded.
        Should lie in range `[-data.ndim - 1, data.ndim]`.
        If `axis < 0`, it is the first axis inserted;
        If `axis >= 0`, it is the last axis inserted in Python's negative indexing.

    num_newaxis : int
        Number of axes to be inserted. Should be >= 0.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    return _make.expand_dims(data, axis, num_newaxis)


def transpose(data, axes=None):
    """Permutes the dimensions of an array.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axes : None or List[int]
        The target axes order, reverse order if not specified.

    Returns
    -------
    result : relay.Expr
        The transposed result.
    """

    if axes is not None:
        axes = list(axes)
    return _make.transpose(data, axes)


def squeeze(data, axis=None):
    """Squeeze axes in the array.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    axis : None or List[int]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    Returns
    -------
    result : tvm.relay.Expr
        The squeezed result.
    """
    return _make.squeeze(data, axis)


def reshape(data, newshape):
    """Reshape the input array.

    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
    The significance of each is explained below:

    ``0`` copy this dimension from the input to the output shape.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
            data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
            data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
            data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    ``-2`` copy all/remainder of the input dimensions to the output shape.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (-2,), result.shape = (2,3,4)
            data.shape = (2,3,4), newshape = (2,-2), result.shape = (2,3,4)
            data.shape = (2,3,4), newshape = (-2,1,1), result.shape = (2,3,4,1,1)

    ``-3`` use the product of two consecutive dimensions of the input shape
    as the output dimension.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
            data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
            data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)
            data.shape = (2,3,4), newshape = (-3,-2), result.shape = (6,4)

    ``-4`` split one dimension of the input into two dimensions passed subsequent
    to -4 in shape (can contain -1).

        .. code-block:: python

            data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape = (1,2,3,4)
            data.shape = (2,3,4), newshape = (2,-4,-1,3,-2), result.shape = (2,1,3,4)

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    newshape : Union[int, Tuple[int], List[int]] or relay.Expr
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    if isinstance(newshape, Expr):
        return _dyn_make.reshape(data, newshape)
    if isinstance(newshape, int):
        newshape = [newshape]
    if isinstance(newshape, (tuple, list)):
        tempshape = []
        for shape in newshape:
            if isinstance(shape, _expr.IntImm):
                tempshape.append(shape.value)
            else:
                try:
                    tempshape.append(int(shape))
                except ValueError as err:
                    raise RuntimeError("Unrecognized shape type: %s" % err)
        newshape = tempshape
    return _make.reshape(data, list(newshape))


def argwhere(condition):
    """Find the indices of elements of a tensor that are
    non-zero.

    Parameters
    ----------
    condition : relay.Expr
        The input condition tensor.

    Returns
    -------
    out : relay.Expr
        Tensor with the indices of elements that are non-zero.

    Examples
    --------
    .. code-block:: python

        condition = [[True, False], [False, True]]
        relay.argwhere(condition) = [[0, 0], [1, 1]]
    """
    return _make.argwhere(condition)


def scatter(data, indices, updates, axis):
    """Update data at positions defined by indices with values in updates

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    axis : int
        The axis to scatter on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.scatter(data, indices, updates, axis)


def scatter_add(data, indices, updates, axis):
    """Update data by adding values in updates at positions defined by indices

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to add.

    axis : int
        The axis to scatter_add on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.scatter_add(data, indices, updates, axis)


def reshape_like(data, shape_like):
    """Reshapes the input array by the size of another array.
    For an input array with shape ``(d1, d2, ..., dk)``, `reshape_like` operation reshapes
    the input array into an output array with the same shape as the second input array.

    .. note::
        Sizes for both array should be compatible.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    shape_like : tuple of int
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.reshape_like(data, shape_like)


def take(data, indices, axis=None, mode="clip"):
    """Take elements from an array along an axis.

    Parameters
    ----------
    data : relay.Expr
        The source array.

    indices : rely.Expr
        The indices of the values to extract.

    axis : int, optional
        The axis over which to select values. By default,
        the flattened input array is used.

    mode : str, optional
        Specifies how out-of-bound indices will behave [clip, wrap, fast].
        clip: clip to the range (default).
        wrap: wrap around the indices.
        fast: no clip or wrap around (user must make sure indices are in-bound).

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.take(data, indices, axis, mode)


def full(fill_value, shape=(), dtype=""):
    """Fill array with scalar value.

    Parameters
    ----------
    fill_value : relay.Expr
        The value to fill. Must be a scalar.

    shape : tuple of int or relay.Expr
        The shape of the target.

    dtype : data type, optional (defaults to data type of the fill value)
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    if isinstance(shape, Expr):
        return _dyn_make.full(fill_value, shape, dtype)
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(shape, (list, tuple)):
        shape = list(shape)
    return _make.full(fill_value, shape, dtype)


def full_like(data, fill_value):
    """Return a scalar value array with the same shape and type as the input array.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    fill_value : relay.Expr
        The scalar value to fill.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.full_like(data, fill_value)


def arange(start, stop=None, step=None, dtype="float32"):
    """Return evenly spaced values within a given interval.

    .. note::
        Similar to ``numpy.arange``, when only one argument is given, it is used
        as `stop` instead of `start` while `start` takes default value 0.

        Warning: Undefined behavior when dtype is incompatible with start/stop/step.
        It could lead to different results compared to numpy, MXNet, pytorch, etc.

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
    result : relay.Expr
        The resulting tensor.

    Examples
    --------
    .. code-block:: python

        relay.arange(5) = [0, 1, 2, 3, 4]
        relay.arange(1, 5) = [1, 2, 3, 4]
        relay.arange(1, 5, 1.5) = [1, 2.5, 4]
    """
    if step is None:
        step = const(1, dtype)

    if stop is None:
        stop = start
        start = const(0, dtype=dtype)

    return _make.arange(start, stop, step, dtype)


def meshgrid(data, indexing="ij"):
    """Create coordinate matrices from coordinate vectors.

    .. note::
        Similar to ``numpy.meshgrid``.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        A list of tensors, which must be either scalars or 1-D vectors.

    indexing : str
        Indexing mode, either "ij" for matrix indexing or "xy" for Cartesian indexing.

    Returns
    -------
    ret : relay.Tuple([relay.Expr, relay.Expr])
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [1, 2, 3]
        y = [4, 5]

        gx, gy = relay.meshgrid([x, y])

        gx = [[1., 1.],
              [2., 2.],
              [3., 3.]]

        gy = [[4., 5.],
              [4., 5.],
              [4., 5.]]
    """
    data = list(data)
    ret_size = len(data)
    return TupleWrapper(_make.meshgrid(Tuple(data), indexing), ret_size)


def repeat(data, repeats, axis):
    """Repeats elements of an array.
    By default, repeat flattens the input array into 1-D and then repeats the elements.

    repeats : int
        The number of repetitions for each element.

    axis: int
        The axis along which to repeat values. The negative numbers are interpreted
        counting from the backward. By default, use the flattened input array, and
        return a flat output array.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        relay.repeat(x, repeats=2) = [1., 1., 2., 2., 3., 3., 4., 4.]

        relay.repeat(x, repeats=2, axis=1) = [[1., 1., 2., 2.],
                                              [3., 3., 4., 4.]]
    """
    return _make.repeat(data, repeats, axis)


def tile(data, reps):
    """Repeats the whole array multiple times.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    reps : tuple of int or relay.Expr
        The number of times repeating the tensor data.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        relay.tile(x, reps=(2,3)) = [[1., 2., 1., 2., 1., 2.],
                                     [3., 4., 3., 4., 3., 4.],
                                     [1., 2., 1., 2., 1., 2.],
                                     [3., 4., 3., 4., 3., 4.]]

        relay.tile(x, reps=(2,)) = [[1., 2., 1., 2.],
                                    [3., 4., 3., 4.]]

    Notes
    -----
    Each dim size of reps must be a positive integer. If reps has length d,
    the result will have dimension of max(d, data.ndim); If data.ndim < d,
    data is promoted to be d-dimensional by prepending new axes.
    If data.ndim >=  d, reps is promoted to a.ndim by pre-pending 1's to it.
    """
    if isinstance(reps, Expr):
        return _dyn_make.tile(data, reps)
    return _make.tile(data, reps)


def reverse(data, axis):
    """Reverses the order of elements along given axis while preserving array shape.
    By default, repeat flattens the input array into 1-D and then repeats the elements.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis: int
        The axis along which to reverse elements.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1., 2.], [3., 4.]]
        relay.reverse(x, axis=0) = [[3., 4.], [1., 2.]]

        relay.reverse(x, axis=1) = [[2., 1.], [4., 3.]]
    """
    return _make.reverse(data, axis)


def reverse_sequence(data, seq_lengths, seq_axis=1, batch_axis=0):
    """Reverse the tensor for variable length slices.
    Input is first sliced along batch axis and then elements are reversed along seq axis.

    Parameters
    ----------
    data : relay.Expr
        The tensor to be reversed.

    seq_lengths : relay.Expr
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
    ret : relay.Expr
        The computed result of same shape and type as of input.

    Examples
    --------
    .. code-block:: python

        x = [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             [12, 13, 14, 15]]
        relay.reverse(x, [1, 2, 3, 4], 0, 1) = [[0, 5, 10, 15],
                                                [4, 1, 6, 11],
                                                [8, 9, 2, 7],
                                                [12, 13, 14, 3]]

        relay.reverse(x, [1, 2, 3, 4], 1, 0) = [[0, 1, 2, 3],
                                                [5, 4, 6, 7],
                                                [10, 9, 8, 11],
                                                [15, 14, 13, 12]]
    """
    return _make.reverse_sequence(data, seq_lengths, seq_axis, batch_axis)


def where(condition, x, y):
    """Selecting elements from either x or y depending on the value of the
    condition.

    .. note::
        The shape of condition, x, and y needs to be the same.

    Parameters
    ----------
    condition : relay.Expr
        The condition array. The n-th element in `y` is selected when the n-th
        value in the `condition` array is zero. Otherwise, the corresponding
        element from `x` will be picked.

    x : relay.Expr
        The first array to be selected.

    y : relay.Expr
        The second array to be selected.

    Returns
    -------
    result : relay.Expr
        The selected array.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        y = [[5, 6], [7, 8]]
        condition = [[0, 1], [-1, 0]]
        relay.where(conditon, x, y) = [[5, 2], [3, 8]]

        condition = [1, 0]
        relay.where(conditon, x, y) = [[1, 2], [7, 8]]
    """
    return _make.where(condition, x, y)


def broadcast_to(data, shape):
    """Return a scalar value array with the same type, broadcast to
    the provided shape.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    shape : tuple of int or relay.Expr
        Provide the shape to broadcast to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    if isinstance(shape, Expr):
        return _dyn_make.broadcast_to(data, shape)
    if isinstance(shape, int):
        shape = [shape]
    if isinstance(shape, (list, tuple)):
        shape = list(shape)
    return _make.broadcast_to(data, shape)


def broadcast_to_like(data, broadcast_type):
    """Return a scalar value array with the same shape and type as the input array.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    broadcast_type : relay.Expr
        Provide the type to broadcast to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.broadcast_to_like(data, broadcast_type)


def collapse_sum_like(data, collapse_type):
    """Return a scalar value array with the same shape and type as the input array.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    collapse_type : relay.Expr
        Provide the type to collapse to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    return _make.collapse_sum_like(data, collapse_type)


def collapse_sum_to(data, shape):
    """Return a summation of data to the specified shape.

    Parameters
    ----------
    data : relay.Expr
        The input tensor.

    shape : relay.Expr
        Shape to collapse to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
    if isinstance(shape, (list, tuple)):
        shape = const(list(shape), "int32")
    return _make.collapse_sum_to(data, shape)


def split(data, indices_or_sections, axis=0):
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis. If such a split is not possible, an error is raised.

    If indices_or_sections is a tuple of sorted integers,
    the entries indicate where along axis the array is split.

    Parameters
    ----------
    data : relay.Expr
        The source array.

    indices_or_sections : int or tuple of int
        Indices or sections to split into. Accepts an int or a tuple

    axis : int, optional
        The axis over which to split.

    Returns
    -------
    ret : relay.Tuple([relay.Expr, relay.Expr])
        The computed result.
    """
    if isinstance(indices_or_sections, int):
        ret_size = indices_or_sections
    else:
        ret_size = len(indices_or_sections) + 1
    return TupleWrapper(_make.split(data, indices_or_sections, axis), ret_size)


def strided_slice(data, begin, end, strides=None, slice_mode="end"):
    """Strided slice of an array.

    Parameters
    ----------
    data : relay.Expr
        The source array to be sliced.

    begin : relay.Expr, Tuple[int], or List[int]
        The indices to begin with in the slicing.

    end : relay.Expr, Tuple[int], or List[int]
        Indices indicating end of the slice.

    strides : relay.Expr, Tuple[int], or List[int], optional
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    slice_mode : str, optional
        The slice mode [end, size].
        end: The ending indices for the slice [default].
        size: The input strides will be ignored, input end in this mode indicates
        the size of a slice starting at the location specified by begin. If end[i]
        is -1, all remaining elements in that dimension are included in the slice.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    strides = strides or [1]
    if isinstance(begin, Expr) or isinstance(end, Expr) or isinstance(strides, Expr):
        if isinstance(begin, (tuple, list)):
            begin = const(list(begin))
        if isinstance(end, (tuple, list)):
            end = const(list(end))
        if isinstance(strides, (tuple, list)):
            strides = const(list(strides))
        normalized_begin = _make.where(
            begin < cast_like(const(0), begin), begin + cast_like(shape_of(data), begin), begin
        )
        return _dyn_make.strided_slice(data, normalized_begin, end, strides, slice_mode)
    return _make.strided_slice(data, begin, end, strides, slice_mode)


def strided_set(data, v, begin, end, strides=None):
    """Strided set of an array.

    Parameters
    ----------
    data : relay.Expr
        The source array to be sliced.

    v : relay.Expr
        The data to be set.

    begin: relay.Expr, Tuple[int], or List[int]
        The indices to begin with in the slicing.

    end: relay.Expr, Tuple[int], or List[int]
        Indices indicating end of the slice.

    strides: relay.Expr, Tuple[int], or List[int], optional
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    strides = strides or const([1], dtype="int32")
    if isinstance(begin, (tuple, list)):
        begin = const(list(begin))
    if isinstance(end, (tuple, list)):
        end = const(list(end))
    if isinstance(strides, (tuple, list)):
        strides = const(list(strides))
    return _make.strided_set(data, v, begin, end, strides)


def slice_like(data, shape_like, axes=None):
    """Slice the first input with respect to the second input.

    For an input array with shape ``(d1, d2, ..., dk)``, `slice_like` operation slices the
    the input array corresponding size of second array. By default will slice on all axes.

    Parameters
    ----------
    data : tvm.relay.Expr
        The source array.

    shape_like : tvm.relay.Expr
        The new shape.

    axes : Optional[Tuple[int]]
        List of axes on which input data will be sliced according to the corresponding size of
        the second input. By default will slice on all axes. Negative axes mean counting in reverse.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.slice_like(data, shape_like, axes)


def layout_transform(data, src_layout, dst_layout):
    """Transform the layout of a tensor

    Parameters
    ----------
    data : relay.Expr
        The source tensor to be transformed

    src_layout: str
        The source layout.  (e.g NCHW)

    dst_layout: str
        The destination layout.  (e.g. NCHW16c)

    Returns
    -------
    ret : relay.Expr
        The transformed tensor.
    """
    return _make.layout_transform(data, src_layout, dst_layout)


def reverse_reshape(data, newshape):
    """Reshapes the input array where the special values are inferred from
    right to left.

    The special values have the same semantics as :py:class:`tvm.relay.reshape`.
    The difference is that special values are inferred from right to left. It
    can be explained in the example below.

    .. code-block:: python

        data.shape = (10,5,4), newshape = (-1,0), reshape results in (40,5)
        data.shape = (10,5,4), newshape = (-1,0), reverse_reshape results in (40,5)

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    newshape : Union[int, Tuple[int], List[int]]
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    if isinstance(newshape, int):
        newshape = [newshape]
    return _make.contrib_reverse_reshape(data, list(newshape))


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
    data: relay.Expr
        The input data to the operator.

    axis: int
        The axis along which to index.

    indices: relay.Expr
        The indices of values to gather.

    Examples
    --------
    .. code-block:: python

        data = [[1, 2], [3, 4]]
        axis = 1
        indices = [[0, 0], [1, 0]]
        relay.gather(data, axis, indices) = [[1, 1], [4, 3]]
    """
    return _make.gather(data, axis, indices)


def gather_nd(data, indices):
    """Gather elements or slices from data and store to a tensor whose shape is
    defined by indices.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The shape of output tensor.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data = [[0, 1], [2, 3]]
        indices = [[1, 1, 0], [0, 1, 0]]
        relay.gather_nd(data, indices) = [2, 3, 0]

        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        indices = [[0, 1], [1, 0]]
        relay.gather_nd(data, indices) = [[3, 4], [5, 6]]
    """
    return _make.gather_nd(data, indices)


def sequence_mask(data, valid_length, mask_value=0, axis=0):
    """Sets all elements outside the expected length of the sequence to a constant value.

    This function takes an n-dimensional input array of the form [MAX_LENGTH, batch_size, ...] or
    [batch_size, MAX_LENGTH, ...] and returns an array of the same shape.

    Parameters
    ----------
    data : relay.Expr
        The input data.

    valid_length : relay.Expr
        The expected (valid) length of each sequence in the tensor.

    mask_value : float
        The masking value.

    axis : int
        The axis of the length dimension.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[[  1.,   2.,   3.], [  4.,   5.,   6.]],
             [[  7.,   8.,   9.], [ 10.,  11.,  12.]],
             [[ 13.,  14.,   15.], [ 16.,  17.,   18.]]]

       relay.sequence_mask(x, valid_length=[1, 1]) =
            [[[  1.,   2.,   3.], [  4.,   5.,   6.]],
             [[  0.,   0.,   0.], [  0.,   0.,   0.]],
             [[  0.,   0.,   0.], [  0.,   0.,   0.]]]

       relay.sequence_mask(x, valid_length=[2, 3], mask_value=0.1) =
            [[[  1.,   2.,   3.], [  4.,   5.,   6.]],
             [[  7.,   8.,   9.], [  10.,  11.,  12.]],
             [[  0.1,  0.1,  0.1], [  16.,  17.,  18.]]]
    """
    return _make.sequence_mask(data, valid_length, mask_value, axis)


def one_hot(indices, on_value, off_value, depth, axis, dtype):
    """
    Returns a one-hot tensor where the locations repsented by indices take value on_value,
    other locations take value off_value.
    Final dimension is <indices outer dimensions> x depth x <indices inner dimensions>.

    Parameters
    ----------
    indices : relay.Expr
        Locations to set to on_value.

    on_value : relay.Expr
        Value to fill at indices.

    off_value : relay.Expr
        Value to fill at all other positions besides indices.

    depth : int or relay.Expr
        Depth of the one-hot dimension.

    axis : int
        Axis to fill.

    dtype : str
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
    if isinstance(depth, Expr):
        return _dyn_make.one_hot(indices, on_value, off_value, depth, axis, dtype)
    return _make.one_hot(indices, on_value, off_value, depth, axis, dtype)


def unravel_index(indices, shape):
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Example::
    -   unravel_index([22, 41, 37], [7, 6]) = [[3, 6, 6],[4, 5, 1]]

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

    return _make.unravel_index(indices, shape)


def sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value=0):
    """Converts a sparse representation into a dense tensor.

    Example::
    -   sparse_to_dense([[0, 0], [1, 1]], [2, 2], [3, 3], 0) = [[3, 0], [0, 3]]

    Parameters
    ----------
    sparse_indices : relay.Expr
        A 0-D, 1-D, or 2-D tensor of integers containing location of sparse values.

    output_shape : relay.Expr
        A list of integers. Shape of the dense output tensor.

    sparse_values : relay.Expr
        A 0-D or 1-D tensor containing the sparse values for the sparse indices.

    default_value : relay.Expr
        A 0-D tensor containing the default value for the remaining locations.
        Defaults to 0.

    Returns
    -------
    result : relay.Expr
        Dense tensor of shape output_shape. Has the same type as sparse_values.
    """

    if default_value == 0:
        default_value = const(0)
    return _make.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)


def matrix_set_diag(data, diagonal):
    """
    Returns a tensor with the diagonal of input tensor replaced with the provided diagonal values.

    Parameters
    ----------
    data : relay.Expr
        Input Tensor.
    diagonal : relay.Expr
        Values to be filled in the diagonal.

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

        relay.matrix_set_diag(input, diagonal) =
            [[[1, 7, 7, 7],
              [7, 2, 7, 7],
              [7, 7, 3, 7]],
             [[4, 7, 7, 7],
              [7, 5, 7, 7],
              [7, 7, 6, 7]]]
    """
    return _make.matrix_set_diag(data, diagonal)


def adv_index(inputs):
    """
    Numpy style advanced indexing. Index with a list of tensors.

    Parameters
    ----------
    inputs : Union(List[relay.Expr], Tuple[relay.Expr])
        Input tensor and indices.
        The first tensor is input data and rests are indices.

    Returns
    -------
    result: relay.Expr
        Output tensor.
    """
    return _make.adv_index(Tuple(inputs))
