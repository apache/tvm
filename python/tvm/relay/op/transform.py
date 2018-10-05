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
