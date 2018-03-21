# pylint: disable=invalid-name,unused-argument,protected-access
"""Overloaded binary operators"""
from __future__ import absolute_import as _abs
import tvm
from . import broadcast as _broadcast


def _tensor_broadcast_bop(lhs, rhs, op):
    """Make an overloaded binary operator of two tensors with auto-broadcasting.

    Parameters
    ----------
    lhs : tvm.Tensor
        Left operand.

    rhs : tvm.Tensor
        Right operand.

    op : str
        Binary operator (e.g., "add", "sub", "mul", "div").

    Returns
    -------
    ret : tvm.Tensor
    """
    if op == "add":
        return _broadcast.broadcast_add(lhs, rhs)
    elif op == "sub":
        return _broadcast.broadcast_sub(lhs, rhs)
    elif op == "mul":
        return _broadcast.broadcast_mul(lhs, rhs)
    elif op == "div":
        return _broadcast.broadcast_div(lhs, rhs)
    else:
        raise ValueError("Broadcast operator %s not supported." % op)


@staticmethod
def _tensor_bop_impl(lhs, rhs, op):
    """Implementation for overloaded binary operator of tensors when applicable;
    raise NotImplementedError if the operator is not supposed to be overloaded.

    Consider the following scenario:
    OP :   + | - | * | /
    R0 :   int | float | Expr | TensorSlice | Tensor (rank zero)
    R1 :   Tensor (positive rank)

    In terms of (LHS OP RHS), we apply the following overloading rules:
    (1) We use broadcast_OP(LHS, RHS) if topi is imported,
        when both LHS and RHS are R1.
    (2) We perform element-wise operation of Tensor and scalar,
        when one of LHS and RHS is R1 and another is R0.
    (3) We do not overload OP (i.e. stick to ExprOp) otherwise.

    Parameters
    ----------
    lhs : object
        Left operand.

    rhs : object
        Right operand.

    op  : str
        Binary operator (e.g., "add", "sub", "mul", "div").

    Returns
    -------
    ret : tvm.Tensor
        Result of overloaded operator.
    """

    def _get_rank(x):
        """Get the rank of a value.
        If x is Tensor, then return its rank;
        if x is scalar (i.e., numeric types, Expr, or TensorSlice), return 0;
        otherwise, return -1.
        """
        if isinstance(x, tvm.tensor.Tensor):
            return len(x.shape)
        elif isinstance(x, (int, float, tvm.expr.Expr, tvm.tensor.TensorSlice)):
            return 0
        return -1

    def _get_func(op):
        """Get function of a given binary operator. """
        if op == "add":
            return lambda x, y: x + y
        elif op == "sub":
            return lambda x, y: x - y
        elif op == "mul":
            return lambda x, y: x * y
        elif op == "div":
            return lambda x, y: x / y
        else:
            raise ValueError("Broadcast operator %s not supported." % op)

    rl = _get_rank(lhs)
    rr = _get_rank(rhs)
    if rl == -1 or rr == -1 or (rl == 0 and rr == 0):
        raise NotImplementedError
    elif rl > 0 and rr > 0:
        return _tensor_broadcast_bop(lhs, rhs, op)
    elif rl == 0:
        f = lambda *i: _get_func(op)(lhs, rhs[tuple(i)])
        return tvm.compute(rhs.shape, f, "tensor_" + op)
    elif rr == 0:
        f = lambda *i: _get_func(op)(lhs[tuple(i)], rhs)
        return tvm.compute(lhs.shape, f, "tensor_" + op)
    else:
        assert False


tvm.tensor.TensorOp._make_bop = _tensor_bop_impl
