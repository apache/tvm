"""Broadcast operators"""
from __future__ import absolute_import as _abs
from .import cpp as _cpp

def broadcast_to(data, shape):
    """Broadcast the src to the target shape

    We follows the numpy broadcasting rule.
    See also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Parameters
    ----------
    data : tvm.Tensor
        The input data

    shape : list or tuple
        The target shape to be broadcasted.

    Returns
    -------
    ret : tvm.Tensor
    """
    return _cpp.broadcast_to(data, shape)


def add(lhs, rhs):
    """Addition with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.add(lhs, rhs)

def bitwise_and(lhs, rhs):
    """Bitwise AND auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.bitwise_and(lhs, rhs)

def bitwise_or(lhs, rhs):
    """Bitwise OR auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.bitwise_or(lhs, rhs)

def bitwise_xor(lhs, rhs):
    """Bitwise XOR auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.bitwise_xor(lhs, rhs)

def bitwise_not(lhs):
    """Bitwise NOT auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if lhs is Expr.
        Otherwise returns Tensor.
    """
    return _cpp.bitwise_not(lhs)

def subtract(lhs, rhs):
    """Subtraction with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """Multiplication with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.multiply(lhs, rhs)


def divide(lhs, rhs):
    """Division with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.divide(lhs, rhs)


def mod(lhs, rhs):
    """Modulus with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.mod(lhs, rhs)


def maximum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.minimum(lhs, rhs)


def power(lhs, rhs):
    """Power with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.power(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.left_shift(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.right_shift(lhs, rhs)


def greater(lhs, rhs):
    """Compute (lhs>rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater(lhs, rhs)


def less(lhs, rhs):
    """Compute (lhs<rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less(lhs, rhs)


def equal(lhs, rhs):
    """Compute (lhs==rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Compute (lhs!=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.not_equal(lhs, rhs)


def greater_equal(lhs, rhs):
    """Compute (lhs>=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater_equal(lhs, rhs)


def less_equal(lhs, rhs):
    """Compute (lhs<=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less_equal(lhs, rhs)
