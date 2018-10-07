"""Transform operators."""

from . import _make


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
        Number of axises to be inserted. Should be >= 0.

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
        The reshaped result.
    """
    axes = axes or []
    return _make.transpose(data, list(axes))


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

    - data.shape = (2,3,4), newshape = (-4,1,2,-2), result.shape =(1,2,3,4)
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
