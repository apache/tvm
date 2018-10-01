"""Generic opertors in TVM.
We follow the numpy naming convention for this interface
(e.g., tvm.generic.multitply ~ numpy.multiply).
The default implementation is used by tvm.ExprOp.
"""
# pylint: disable=unused-argument
from . import make as _make

#Operator precedence used when overloading.
__op_priority__ = 0

def add(lhs, rhs):
    """Generic add operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of add operaton.
    """
    return _make._OpAdd(lhs, rhs)


def subtract(lhs, rhs):
    """Generic subtract operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of subtract operaton.
    """
    return _make._OpSub(lhs, rhs)


def multiply(lhs, rhs):
    """Generic multiply operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of multiply operaton.
    """
    return _make._OpMul(lhs, rhs)


def divide(lhs, rhs):
    """Generic divide operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make._OpDiv(lhs, rhs)


def cast(src, dtype):
    """Generic cast operator.

    Parameters
    ----------
    src : object
        The source operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make.static_cast(dtype, src)
