"""Reduce operators."""
# pylint: disable=redefined-builtin

from . import _make

def argmax(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of maximum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """

    return _make.argmax(data, axis, keepdims, exclude)

def argmin(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """

    return _make.argmin(data, axis, keepdims, exclude)
