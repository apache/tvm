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
    return _make.equal(lhs, rhs)


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
