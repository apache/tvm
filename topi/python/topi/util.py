"""Common topi utilities"""
from __future__ import absolute_import as _abs
import tvm

def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
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
        if not isinstance(elem, (tvm.expr.IntImm, tvm.expr.UIntImm)):
            raise ValueError("Element of input tuple should be const int")
        out_tuple = out_tuple + (elem.value, )
    return out_tuple
