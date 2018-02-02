"""Common topi utilities"""
from __future__ import absolute_import as _abs
import tvm

def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, int):
        return expr
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        raise ValueError("Expect value to be constant int")
    return expr.value


def equal_const_int(expr, value):
    """Returns if expr equals value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, int):
        return expr == value
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return False
    return expr.value == value


def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    out_tuple = ()
    for elem in in_tuple:
        value = get_const_int(elem)
        out_tuple = out_tuple + (value, )
    return out_tuple


def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.

    Parameters
    ----------
    expr : Expr or int
        The input.

    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr


def ravel_index(indices, shape):
    """Flatten the index tuple to 1D

    Parameters
    ----------
    indices : tuple of int or tvm.expr.IntImm
        The input coordinates

    shape : tuple of int
        Shape of the tensor.

    Returns
    -------
    idx : int or Expr
        The index after flattening
    """
    idx = None
    for i, (shape_val, ind) in enumerate(zip(shape, indices)):
        if i != 0:
            idx = idx * shape_val + ind
        else:
            idx = ind
    return idx


def unravel_index(idx, shape):
    """Convert the flattened ind to the coordinate array

    Parameters
    ----------
    idx : int or tvm.expr.IntImm
        The 1D index

    shape : tuple of int
        Shape of the tensor

    Returns
    -------
    indices : tuple of int or tvm.expr.IntImm
        Corresponding coordinate of the 1D index
    """
    indices = []
    for i in range(len(shape) - 1, -1, -1):
        indices.append(idx % shape[i])
        idx = idx // shape[i]
    indices = indices[::-1]
    return indices
