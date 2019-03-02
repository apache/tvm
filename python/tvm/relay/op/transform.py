"""Transform operators."""

from . import _make
from ..expr import TupleWrapper


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
    from .. import _make as _relay_make
    return _relay_make.cast(data, dtype)


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
    """Reshapes the input array.

    Example::

    To give user more convenience in without doing manual shape inference,
    some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}.
    The significance of each is explained below:

    - ``0``  copy this dimension from the input to the output shape.

    Example::

    - data.shape = (2,3,4), newshape = (4,0,2), result.shape = (4,3,2)
    - data.shape = (2,3,4), newshape = (2,0,0), result.shape = (2,3,4)

    - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
    keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

    Example::

    - data.shape = (2,3,4), newshape = (6,1,-1), result.shape = (6,1,4)
    - data.shape = (2,3,4), newshape = (3,-1,8), result.shape = (3,1,8)
    - data.shape = (2,3,4), newshape = (-1,), result.shape = (24,)

    - ``-2`` copy all/remainder of the input dimensions to the output shape.

    Example::

    - data.shape = (2,3,4), newshape = (-2,), result.shape = (2,3,4)
    - data.shape = (2,3,4), newshape = (2,-2), result.shape = (2,3,4)
    - data.shape = (2,3,4), newshape = (-2,1,1), result.shape = (2,3,4,1,1)

    - ``-3`` use the product of two consecutive dimensions of the input shape
    as the output dimension.

    Example::

    - data.shape = (2,3,4), newshape = (-3,4), result.shape = (6,4)
    - data.shape = (2,3,4,5), newshape = (-3,-3), result.shape = (6,20)
    - data.shape = (2,3,4), newshape = (0,-3), result.shape = (2,12)
    - data.shape = (2,3,4), newshape = (-3,-2), result.shape = (6,4)

    - ``-4`` split one dimension of the input into two dimensions passed subsequent
    to -4 in shape (can contain -1).

    Example::

    - data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape = (1,2,3,4)
    - data.shape = (2,3,4), newshape = (2,-4,-1,3,-2), result.shape = (2,1,3,4)

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
    return _make.reshape(data, list(newshape))


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


def take(data, indices, axis=None):
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

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.take(data, indices, axis)


def full(fill_value, shape=(), dtype=""):
    """Fill array with scalar value.

    Parameters
    ----------
    fill_value : relay.Expr
        The value to fill. Must be a scalar.

    shape : tuple of int
        The shape of the target.

    dtype : data type, optional (defaults to data type of the fill value)
        The data type of the target.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
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


def arange(start, stop=None, step=1, dtype="float32"):
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
    if stop is None:
        stop = start
        start = 0
    return _make.arange(start, stop, step, dtype)


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

    shape : shape
        Provide the shape to broadcast to.

    Returns
    -------
    result : relay.Expr
        The resulting tensor.
    """
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


def strided_slice(data, begin, end, strides=None):
    """Strided slice of an array.

    Parameters
    ----------
    data : relay.Expr
        The source array to be sliced.

    begin: list of int
        The indices to begin with in the slicing.

    end: list of int
        Indicies indicating end of the slice.

    strides: list of int, optional
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    strides = strides or []
    return _make.strided_slice(data, list(begin), list(end), list(strides))


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

    Example::

    The special values have the same semantics as :py:class:`tvm.relay.reshape`.
    The difference is that special values are inferred from right to left. It
    can be explained in the example below::

    - data.shape = (10,5,4), newshape = (-1,0), reshape results in (40,5)
    - data.shape = (10,5,4), newshape = (-1,0), reverse_reshape results in (40,5)

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
    return _make._contrib_reverse_reshape(data, list(newshape))
