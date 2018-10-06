"""Basic tensor operations."""
from __future__ import absolute_import as _abs
from . import _make
from ..expr import Tuple

# We create a wrapper function for each operator in the
# python side to call into the positional _make.OpName function.
#
# We make this decision so that we can:
# - Have declare python docstring for each function
# - Enable keyword arguments easily
# - Not put too much burden on FFI to support complicated features
#   like default value and keyword arguments

def log(data):
    """Compute elementwise log of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.log(data)


def exp(data):
    """Compute elementwise exp of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.exp(data)


def sqrt(data):
    """Compute elementwise sqrt of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sqrt(data)


def sigmoid(data):
    """Compute elementwise sigmoid of data.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.sigmoid(data)


def add(lhs, rhs):
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code:: python

      x = relay.Var("a") # shape is [2, 3]
      y = relay.Var("b") # shape is [2, 1]
      z = relay.add(x, y)  # result shape is [2, 3]
    """
    return _make.add(lhs, rhs)


def subtract(lhs, rhs):
    """Elementwise subtraction with broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.subtract(lhs, rhs)


def equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs == rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs != rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.not_equal(lhs, rhs)


def less(lhs, rhs):
    """Broadcasted elementwise test for (lhs < rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less(lhs, rhs)


def less_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs <= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.less_equal(lhs, rhs)


def greater(lhs, rhs):
    """Broadcasted elementwise test for (lhs > rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater(lhs, rhs)


def greater_equal(lhs, rhs):
    """Broadcasted elementwise test for (lhs >= rhs).

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.greater_equal(lhs, rhs)


def maximum(lhs, rhs):
    """Maximum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Minimum with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.minimum(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.right_shift(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side input data
    rhs : relay.Expr
        The right hand side input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.left_shift(lhs, rhs)


def concat(*args):
    """Concatenate the input tensors along the zero axis.

    Parameters
    ----------
    args: list of Tensor

    Returns
    -------
    tensor: The concatenated tensor.
    """
    tup = Tuple(list(args))
    return _make.concat(tup)


def zeros_like(data):
    """Returns an array of zeros, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.zeros_like(data)


def ones_like(data):
    """Returns an array of ones, with same type and shape as the input.

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.ones_like(data)

def clip(a, a_min, a_max):
    """Clip the elements in `a` between `a_min` and `a_max`. `a_min` and `a_max` are cast to `a`'s dtype.

    Parameters
    ----------
    a : relay.Expr
        The input tensor.
    a_min : float
        The clip minimum.
    a_max : float
        The clip maximum.

    Returns
    -------
    result : relay.Expr
        `a` with elements clipped between `a_min` and `a_max`.

    Examples
    --------
    .. code:: python
      x = relay.Constant(tvm.nd.array([0, 1, 5, 3, 4, 2]))
      relay.clip(x, 1., 4.)
      # [1, 1, 4, 3, 4, 2]
    """
    return _make.clip(a, a_min, a_max)
