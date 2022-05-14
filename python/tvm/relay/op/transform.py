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

from ...tir import expr as _expr
from ..expr import Constant, Expr, Tuple, TupleWrapper, const
from . import _make
from .dyn import _make as _dyn_make
from .tensor import shape_of


def sliding_window(data, axis, window_shape, strides):
    """Slide a window over the data tensor.

    Parameters
    ----------
    data : relay.Expr
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
    result : relay.Expr
        The resulting tensor.

    Examples
    --------
    .. code-block:: python

        # Slide a window of shape (3, 4, 5) over the x tensor, beginning with
        # dimension 1, which slides the window over the two subtensors of
        # shape (3, 32, 32).
        x = relay.var("x", relay.TensorType((2, 3, 32, 32), "float32"))
        y = relay.sliding_window(x, 1, [3, 4, 5], [1, 2, 3])

        data = np.random.rand(2, 3, 32, 32).astype("float32")
        result = create_executor().evaluate(y, {x: relay.const(data)}).numpy()

        # The resulting shape still has batch size 2. Each dimension in
        # (1, 15, 10) represents the locations where we were able to
        # form a window; that is, we were able to place the window
        # in one place along the dimension of length 3, 15 places along
        # the dimension of length 32 (when striding by 2), and 10 places
        # along the second dimension of length 32 (when striding by 3).
        # The remaining dimension (3, 4, 5) represent the formed windows.
        assert result.shape == (2, 1, 15, 10, 3, 4, 5)

        assert np.array_equal(result[0, 0, 0, 0, :, :, :], data[0, :, 0:4, 0:5])
        assert np.array_equal(result[1, 0, 7, 3, :, :, :], data[1, :, 14:18, 9:14])
        assert np.array_equal(result[1, 0, 14, 9, :, :, :], data[1, :, 28:32, 27:32])
    """
    from .. import _ffi_api as _relay_make

    return _relay_make.sliding_window(data, axis, window_shape, strides)


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
    """Insert `num_newaxis` axes at the position given by `axis`.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : Union[int, Expr]
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
    if isinstance(axis, int):
        return _make.expand_dims(data, axis, num_newaxis)
    if isinstance(axis, Expr):
        # TODO (AndrewZhaoLuo): investigate performance issues with consecutive
        # dynamic expand_dims on non-llvm targets.
        return _dyn_make.expand_dims(data, axis, num_newaxis)
    raise ValueError(f"Unknown type for axis: {type(axis)}")


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

    axis : None or List[int] or Expr
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    Returns
    -------
    result : tvm.relay.Expr
        The squeezed result.
    """
    if isinstance(axis, Constant):
        axis = list(axis.data.numpy())
    if isinstance(axis, Expr):
        return _dyn_make.squeeze(data, axis)
    return _make.squeeze(data, axis)


def reshape(data, newshape, allowzero=False):
    """Reshape the input array.

    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
    The significance of each is explained below:

    ``0`` copy this dimension from the input to the output shape.

        .. code-block:: python

            data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
            data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    Note: If the parameter allowzero is manually set to true, it specifies a
    special case where 0 actually means a true empty tensor.

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

    allowzero : Bool, optional
        If true, then treat zero as true empty tensor rather than a copy instruction.

    Returns
    -------
    result : relay.Expr
        The reshaped result.
    """
    if isinstance(newshape, Constant):
        newshape = list(newshape.data.numpy())
    if isinstance(newshape, Expr):
        return _dyn_make.reshape(data, newshape, allowzero)
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
    return _make.reshape(data, list(newshape), allowzero)


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


def scatter_nd(data, indices, updates, mode="update"):
    """Scatter values from an array and update.

    See :py:func:`tvm.topi.scatter` for how data is scattered.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    mode : string
        The accumulation mode for scatter. "update" or "add"

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.scatter_nd(data, indices, updates, mode)


def reshape_like(data, shape_like, lhs_begin=0, lhs_end=None, rhs_begin=0, rhs_end=None):
    """Reshapes the input tensor by the size of another tensor.
    For an input tensor with shape ``(d0, d1, ..., d(k-1))``, `reshape_like` operation reshapes
    the input tensor into an output tensor with the same shape as the second input tensor,
    in particular reshaping the dimensions of `data` in `[lhs_begin, lhs_end)` using the dimensions
    from `shape_like` in `[rhs_begin, rhs_end)`.

    .. note::
        Sizes for `data` and the output tensor should be compatible.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    shape_like : relay.Expr
        The tensor to reshape data like. Should be compatible with the original shape on the
        reshaped dimensions.

    lhs_begin : int, optional
        The axis of data to begin reshaping. Default is 0.

    lhs_end : int or None, optional
        The axis of data where reshaping should stop, exclusive. Default is None which reshapes to
        the end.

    rhs_begin : int, optional
        The axis of shape_like where the target shape begins. Default is 0.

    rhs_end : int or None, optional
        The axis of shape_like where the target shape ends, exclusive. Default is None which extends
        to the end.

    Returns
    -------
    ret : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data.shape == (1, 2, 3, 4)
        shape_like.shape == (6, 2, 2, 3)

        ret = relay.reshape_like(data, shape_like, lhs_begin=1, rhs_end=3)
        ret.shape == (1, 6, 2, 2)
    """
    return _make.reshape_like(data, shape_like, lhs_begin, lhs_end, rhs_begin, rhs_end)


def take(data, indices, axis=None, batch_dims=0, mode="clip"):
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

    batch_dims : int
        The number of batch dimensions. By default is 0.

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
    return _make.take(data, indices, batch_dims, axis, mode)


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
    if isinstance(shape, Constant):
        shape = list(shape.data.numpy())
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
    if isinstance(reps, Constant):
        reps = list(reps.data.numpy())
    if isinstance(reps, Expr):
        return _dyn_make.tile(data, reps)
    return _make.tile(data, reps)


def reverse(data, axis):
    """Reverses the order of elements along given axis while preserving array shape.

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
        Shapes of condition, x, and y must be broadcastable to a common shape.
        Semantics follow numpy where function
        https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Parameters
    ----------
    condition : relay.Expr
        Where True, yield x, otherwise yield y

    x : relay.Expr
        The first array or scalar to be selected.

    y : relay.Expr
        The second array or scalar to be selected.

    Returns
    -------
    result : relay.Expr
        The selected array. The output shape is the broadcasted shape from
        condition, x, and y.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        y = [[5, 6], [7, 8]]
        condition = [[0, 1], [-1, 0]]
        relay.where(conditon, x, y) = [[5, 2], [3, 8]]

        condition = [[1], [0]]
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
    if isinstance(shape, Constant):
        shape = list(shape.data.numpy())
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


def strided_slice(data, begin, end, strides=None, axes=None, slice_mode="end"):
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

    axes : Tuple[int] or List[int], optional
        Axes along which slicing is applied. When it is specified, the length of begin, end,
        strides, and axes must be equal. Moreover, begin, end, strides, and axes must be
        static (cannot be relay.Expr). Axes argument for dynamic parameter slicing is
        not supported yet.

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
    if isinstance(begin, Constant):
        begin = list(begin.data.numpy())
    if isinstance(end, Constant):
        end = list(end.data.numpy())
    if isinstance(strides, Constant):
        strides = list(strides.data.numpy())
    if isinstance(begin, Expr) or isinstance(end, Expr) or isinstance(strides, Expr):
        if isinstance(begin, (tuple, list)):
            begin = const(list(begin))
        if isinstance(end, (tuple, list)):
            end = const(list(end))
        if isinstance(strides, (tuple, list)):
            strides = const(list(strides))

        ishape = cast_like(shape_of(data), begin)
        ishape_slice = slice_like(ishape, begin)
        begin = _make.where(begin < cast_like(const(0), begin), begin + ishape_slice, begin)
        begin = _make.where(begin >= ishape_slice, ishape_slice, begin)
        # TODO(masahi): Support axes argument in dynamic strided slice
        assert axes is None, "Axes argument for dynamic parameter slicing is not supported yet."
        return _dyn_make.strided_slice(data, begin, end, strides, slice_mode)
    return _make.strided_slice(data, begin, end, strides, slice_mode, axes)


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
        The axis along which to index. negative axis is supported.

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


def gather_nd(data, indices, batch_dims=0, index_rank=None):
    """Gather elements or slices from data and store to a tensor whose shape is
    defined by indices.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The shape of output tensor.

    batch_dims : int
        The number of batch dimensions.

    index_rank : int, optional
        The size of an indexing tuple, which is a fixed value and the same as indices.shape[0]
        Only needed when other dimensions of indices are dynamic.

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

        data    = [[[0,1],[2,3]],[[4,5],[6,7]]]
        indices = [[1, 0]]
        relay.gather_nd(data, indices, batch_dims=1) = [[2,3],[4,5]]
    """
    return _make.gather_nd(data, indices, batch_dims, index_rank)


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
    if isinstance(depth, Constant):
        depth = depth.data.numpy().item()
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
    if isinstance(output_shape, Expr):
        return _dyn_make.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)
    return _make.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value)


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

        relay.matrix_set_diag(input, diagonal) =
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

    return _make.matrix_set_diag(
        data, diagonal, k_one, k_two, super_diag_right_align, sub_diag_right_align
    )


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


def sparse_fill_empty_rows(sparse_indices, sparse_values, dense_shape, default_value):
    """
    Fill rows in a sparse matrix that do no contain any values. Values are placed in the first
    column of empty rows. The sparse array is in COO format.
    It returns a TupleWrapper with 3 outputs

    Parameters
    ----------
    sparse_indices : relay.Expr
        A 2-D tensor[N, ndims] of integers containing location of sparse values, where N is
        the number of sparse values and n_dim is the number of dimensions of the dense_shape.
        The first column of this relay parameter must be sorted in ascending order.

    sparse_values : relay.Expr
        A 1-D tensor[N] containing the sparse values for the sparse indices.

    dense_shape : relay.Expr
        A 1-D tensor[ndims] which contains shape of the dense output tensor.

    default_value : relay.Expr
        A 1-D tensor[1] containing the default value for the remaining locations.

    Returns
    -------
    new_sparse_indices : relay.Expr
        A 2-D tensor[?, ndims] of integers containing location of new sparse
        indices. The first column outputs must be sorted in ascending order.

    new_sparse_values : relay.Expr
        A 1-D tensor[?] containing the sparse values for the sparse indices.

    empty_row_indicator : relay.Expr
        A 1-D tensor[dense_shape[0]] filled with zeros and ones
        indicating whether the particular row is empty or full respectively

    Note
    ----
    This op exactly follows the documentation here:
    https://www.tensorflow.org/api_docs/python/tf/sparse/fill_empty_rows
    There are two exceptions:
    1. Input Sparse Indices are expected to be in row-major order.
    2. Empty Row Indicator has int64 output type with 1(for True) and 0(for False).

    Examples
    -------
    .. code-block:: python

        sparse_indices = [[0, 1],
                         [0, 3],
                         [2, 0],
                         [3, 1]]
        sparse_values = [1, 2, 3, 4]
        default_value = [10]
        dense_shape = [5, 6]
        new_sparse_indices, empty_row_indicator, new_sparse_values, slice_element_index =
                            relay.sparse_fill_empty_rows(
                            sparse_indices,
                            sparse_values,
                            default_value,
                            dense_shape)
        new_sparse_indices = [[0, 1],
                             [0, 3],
                             [1, 0],
                             [2, 0],
                             [3, 1],
                             [4, 0]]
        empty_row_indicator = [False, True, False, False, True]
        new_sparse_values = [1, 2, 10, 3, 4, 10]
    """
    new_sparse_indices, new_sparse_values, empty_row_indicator = TupleWrapper(
        _make.sparse_fill_empty_rows(sparse_indices, sparse_values, dense_shape, default_value), 3
    )
    new_sparse_indices = cast_like(new_sparse_indices, sparse_indices)
    new_sparse_values = cast_like(new_sparse_values, sparse_values)
    empty_row_indicator = cast(empty_row_indicator, "bool")

    return Tuple((new_sparse_indices, new_sparse_values, empty_row_indicator))


def sparse_reshape(sparse_indices, prev_shape, new_shape):
    """
    Reshape a Sparse Tensor. The sparse array is in COO format.

    Parameters
    ----------
    sparse_indices : relay.Expr
        A 2-D tensor[N, n_dim] of integers containing location of sparse values, where N is the
        number of sparse values and n_dim is the number of dimensions of the dense_shape
    prev_shape : relay.Expr
        A 1-D tensor containing the previous shape of the dense tensor
    new_shape : relay.Expr
        A 1-D tensor containing the new shape of the dense tensor
    Returns
    -------
    result: relay.Expr
        Output tensor.
    Examples
    --------
    .. code-block:: python

        sparse_indices = [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 2, 3]]
        prev_shape = [2, 3, 4]
        new_shape = [9, -1]
        new_sparse_indices, new_shape = relay.sparse_reshape(sparse_indices,
                            prev_shape,
                            new_shape)
        new_sparse_indices = [[0, 0],
                              [0, 1],
                              [1, 2],
                              [4, 2],
                              [8, 1]]
        new_shape = [9, 4]
    """
    return TupleWrapper(_make.sparse_reshape(sparse_indices, prev_shape, new_shape), 2)


def segment_sum(data, segment_ids, num_segments=None):
    """
    Computes the sum along segment_ids along axis 0. If multiple segment_ids reference the same
    location their contributions add up.
    result[index, j, k, ...] = Σi... data[i, j, k,..] where index = segment_ids[i]
    This op is much better understood with visualization articulated in the following links and
    examples at the end of this docstring.

    https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum
    https://caffe2.ai/docs/sparse-operations.html#null__unsorted-segment-reduction-ops

    Parameters
    ----------
    data : relay.Expr
        Input Tensor. It can be of any type and multi-dimensional
    segment_ids : relay.Expr
        A 1-D int32/int64 tensor containing the segment_ids of the rows to calculate the output
        sum upon. It defines a mapping from the zeroth dimension of data onto segment_ids. The
        segment_ids tensor should be the size of the first dimension, d0, with consecutive IDs
        in the range 0 to k, where k<d0. In particular, a segmentation of a matrix tensor is a
        mapping of rows to segments. This tensor doesn't need to be sorted
    num_segments : Optional[int]
        An integer describing the shape of the zeroth dimension. If unspecified, its calculated
        equivalent to the number of unique segment_ids
    Returns
    -------
    result: relay.Expr
        Output tensor.
    Examples
    --------
    .. code-block:: python

        data = [[1, 2, 3, 4],
                [4, -3, 2, -1],
                [5, 6, 7, 8]]
        segment_ids = [0, 0, 1]
        result = segment_sum(data, segment_ids)
        result = [[5, -1, 5, 3],[5, 6, 7, 8]]

        data = [[1, 2, 3, 4],
                [4, -3, 2, -1],
                [5, 6, 7, 8]]
        segment_ids = [2, 0, 0]
        num_segments = 3
        result = segment_sum(data, segment_ids, num_segments)
        result = [[5, 6, 7, 8],[0, 0, 0, 0], [5, -1, 5, 3]]
    """

    one_tensor = cast_like(const([1]), segment_ids)
    if num_segments:
        if isinstance(num_segments, int):
            max_segments = const([num_segments])
            max_segments = cast_like(max_segments, segment_ids)
        else:
            max_segments = cast_like(num_segments, segment_ids)
    else:
        max_segments = _make.add(reshape(_make.max(segment_ids, [0], False, False), -1), one_tensor)

    data_offrow_shape = strided_slice(_make.shape_of(data, "int32"), [1], [-1], slice_mode="size")
    data_offrow_shape = cast_like(data_offrow_shape, max_segments)
    new_shape = _make.concatenate(Tuple([max_segments, data_offrow_shape]), 0)
    segment_ids_tiled_shape = _make.concatenate(
        Tuple([reverse(data_offrow_shape, 0), one_tensor]), 0
    )
    expanded_segment_ids = tile(segment_ids, segment_ids_tiled_shape)
    scatter_add_segment_ids = transpose(expanded_segment_ids)
    src = cast_like(_dyn_make.zeros(new_shape, "float64"), data)
    return scatter_add(src, scatter_add_segment_ids, data, axis=0)


def cumsum(data, axis=None, dtype=None, exclusive=None):
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If true will return exclusive sum in which the first element is not
        included. In other terms, if true, the j-th output element would be
        the sum of the first (j-1) elements. Otherwise, it would be the sum of
        the first j elements.

    Returns
    -------
    result : relay.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1,2,3], [4,5,6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, dtype="float32")
        -> [  1.,   3.,   6.,  10.,  15.,  21.]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]

        a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array
        cumsum(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 2, 2, 3, 4, 4]
    """
    return _make.cumsum(data, axis, dtype, exclusive)


def cumprod(data, axis=None, dtype=None, exclusive=None):
    """Numpy style cumprod op. Return the cumulative inclusive product of the elements along
    a given axis.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None) is to compute
        the cumprod over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are multiplied.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If true will return exclusive product in which the first element is not
        included. In other terms, if true, the j-th output element would be
        the product of the first (j-1) elements. Otherwise, it would be the product of
        the first j elements. The product of zero elements will be 1.

    Returns
    -------
    result : relay.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1,2,3], [4,5,6]]

        cumprod(a)  # if axis is not provided, cumprod is done over the flattened input.
        -> [ 1,  2,  6, 24, 120, 720]

        cumprod(a, dtype="float32")
        -> [  1.,  2.,  6., 24., 120., 720.]

        cumprod(a, axis=0)  # multiply over rows for each of the 3 columns
        -> [[1, 2, 3],
            [4, 10, 18]]

        cumprod(a, axis=1)
        -> [[ 1,  2,  6],
            [ 4,  20, 120]]

        a = [1, 1, 1, 0, 1, 1, 0]  # a is a boolean array
        cumprod(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 1, 0, 0, 0, 0]
    """
    return _make.cumprod(data, axis, dtype, exclusive)


def unique(data, is_sorted=True, return_counts=False):
    """
    Find the unique elements of a 1-D tensor. Please note `output` and `counts` are all padded to
    have the same length of `data` and element with index >= num_unique[0] has undefined value.

    Parameters
    ----------
    data : relay.Expr
        A 1-D tensor of integers.

    is_sorted : bool
        Whether to sort the unique elements in ascending order before returning as output.

    return_counts : bool
        Whether to return the count of each unique element.

    Returns
    -------
    unique : relay.Expr
        A 1-D tensor containing the unique elements of the input data tensor.

    indices : relay.Expr
        A 1-D tensor containing the index of each data element in the output tensor.

    inverse_indices : relay.Expr
        A 1-D tensor. For each entry in data, it contains the index of that data element in the
        unique array.

    num_unique : relay.Expr
        A 1-D tensor with size=1 containing the number of unique elements in the input data tensor.

    counts (optional) : relay.Expr
        A 1-D tensor containing the count of each unique element in the output.

    Examples
    --------
    .. code-block:: python

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, False)
        output         =  [4, 5, 1, 2, 3, _, _, _]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]

        [output, indices, num_unique, counts] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, True)
        output         =  [4, 5, 1, 2, 3, _, _, _]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]
        counts         =  [2, 2, 1, 1, 2, _, _, _]

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], True)
        output         =  [1, 2, 3, 4, 5, _, _, _]
        indices        =  [3, 4, 0, 1, 2, 2, 3, 4]
        num_unique     =  [5]
    """
    if return_counts:
        return TupleWrapper(_make.unique(data, is_sorted, return_counts), 5)
    return TupleWrapper(_make.unique(data, is_sorted, return_counts), 4)


def invert_permutation(data):
    """Computes the inverse permutation of data.
    This operation computes the inverse of an index permutation.
    It takes a 1-D integer tensor x, which represents the indices of a zero-based
    array and swaps each value with its index position.

    For an output tensor y and an input tensor x, this operation computes the following:
    y[x[i]] = i for i in [0, 1, ..., len(x) - 1]

    Parameters
    ----------
    data : relay.Expr
        The source data to be invert permuated.

    Returns
    -------
    ret : relay.Expr
        Invert permuated data. Has the same type as data.

    Examples
    --------
    .. code-block:: python

        data = [3, 4, 0, 2, 1]
        relay.invert_permutation(data) = [2, 4, 3, 0, 1]
    """
    return _make.invert_permutation(data)


def stft(
    data, n_fft, hop_length=None, win_length=None, window=None, normalized=False, onesided=True
):
    """
    The STFT computes the Fourier transform of short overlapping windows of the input.
    This gives frequency components of the signal as they change over time.

    Parameters
    ----------
    data : relay.Expr
        Either a 1-D tensor or a 2-D batch tensor.

    n_fft : int
        The size of Fourier transform

    hop_length : int, optional
        The distance between neighboring sliding window frames. If is None,
        it is treated as equal to floor(n_fft / 4).

    win_length : int, optional
        The size of window frame and STFT filter. If is None, it is treated as equal to n_fft.

    window : relay.Expr, optional
        A 1-D tensor window frame. If is None (default), it is treated as if
        having 1 everywhere in the window.

    normalized : bool, optional
        Whether to return the normalized STFT results. Default value is False.

    onesided : bool, optional
        Whether to return onesided result or fill with conjugate symmetry. Default value is True.

    Returns
    -------
    output : relay.Expr
        Tensor containing the STFT result with shape [batch, N, T, 2], where N is the
        number of frequencies where STFT is applied and T is the total number of frames used.

    Examples
    --------
    .. code-block:: python

        data = [1, 2, 3, 4, 5, 6]
        window = [4, 3, 2]
        [n_fft, hop_length, win_length, normalized, onesided] = [3, 3, 3, False, True]
        relay.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)
        -> [[[15.0000,  0.0000], [34.0000,  0.0000]], [[ 4.5000,  0.8660], [ 1.0000, -1.7321]]]
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if win_length is None:
        win_length = n_fft

    if window is None:
        window = _make.ones([n_fft], "int32")

    return _make.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)
