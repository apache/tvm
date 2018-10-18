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


def sum(data, axis=None, keepdims=False, exclude=False):
    """Computes the sum of array elements over given axes.

    Example::

      data = [[[1,2],[2,3],[1,3]],
              [[1,4],[4,3],[5,2]],
              [[7,1],[7,2],[7,3]]]

      sum(data, axis=1)
      [[  4.   8.]
       [ 10.   9.]
       [ 21.   6.]]

      sum(data, axis=[1,2])
      [ 12.  19.  27.]

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

    return _make.sum(data, axis, keepdims, exclude)


def max(data, axis=None, keepdims=False, exclude=False):
    """ Computes the max of array elements over given axes.

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

    return _make.max(data, axis, keepdims, exclude)


def min(data, axis=None, keepdims=False, exclude=False):
    """Computes the min of array elements over given axes.

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

    return _make.min(data, axis, keepdims, exclude)


def mean(data, axis=None, keepdims=False, exclude=False):
    """Computes the mean of array elements over given axes.

    Example::

      data = [[[1,2],[2,3],[1,3]],
              [[1,4],[4,3],[5,2]],
              [[7,1],[7,2],[7,3]]]

      mean(data)
      [3.22]

      mean(data, axis=[1,2])
      [ 2.  3.16666667  4.5]

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

    return _make.mean(data, axis, keepdims, exclude)


def prod(data, axis=None, keepdims=False, exclude=False):
    """Computes the products of array elements over given axes.

    Example::

      data = [[[1,2],[2,3],[1,3]],
              [[1,4],[4,3],[5,2]],
              [[7,1],[7,2],[7,3]]]

      mean(data, axis=1)
      [35562240]

      mean(data, axis=[1,2])
      [ 36  480  2058]

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

    return _make.prod(data, axis, keepdims, exclude)
