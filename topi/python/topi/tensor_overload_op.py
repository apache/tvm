# pylint: disable=invalid-name,unused-argument,protected-access
"""Overloaded binary operators"""
from __future__ import absolute_import as _abs
import tvm
from . import broadcast as _broadcast


def _tensor_bop_impl(lhs, rhs, name, elementwise_bop, broadcast_bop):
    """Implementation for overloaded binary operator of tensors when applicable;
    raise NotImplementedError if the operator is not supposed to be overloaded.

    Consider the following scenario:
    OP :   + | - | * | /
    R0 :   int | float | Expr | TensorSlice | Tensor (rank zero)
    R1 :   Tensor (positive rank)

    In terms of (LHS OP RHS), we apply the following overloading rules:
    (1) We use broadcast_OP(LHS, RHS), when both LHS and RHS are R1.
    (2) We perform element-wise operation of Tensor and scalar,
        when one of LHS and RHS is R1 and another is R0.
    (3) We do not overload OP (i.e. stick to ExprOp) otherwise.

    Parameters
    ----------
    lhs : object
        Left operand.

    rhs : object
        Right operand.

    name  : str
        Binary operator name, used for naming tensor-scalar computation.

    elementwise_bop : function
        Binary operator for element-wise tensor-scalar operation for rule (2).

    broadcast_bop : function
        Binary operator for broadcast tensor-tensor operation for rule (1).

    Returns
    -------
    ret : tvm.Tensor
        Implementation of overloaded operator.
    """

    def _get_rank(x):
        """Get the rank of a value.
        If x is Tensor, then return its rank;
        if x is scalar_like (i.e., numeric types, Expr, or TensorSlice), return 0;
        otherwise, return -1.
        """
        if isinstance(x, tvm.tensor.Tensor):
            return len(x.shape)
        elif isinstance(x, (int, float, tvm.expr.Expr, tvm.tensor.TensorSlice)):
            return 0
        return -1

    rl = _get_rank(lhs)
    rr = _get_rank(rhs)
    if rl == -1 or rr == -1 or (rl == 0 and rr == 0):
        raise NotImplementedError
    elif rl > 0 and rr > 0:
        return broadcast_bop(lhs, rhs)
    elif rl == 0:
        f = lambda *i: elementwise_bop(lhs, rhs[tuple(i)])
        return tvm.compute(rhs.shape, f, "tensor_" + name)
    elif rr == 0:
        f = lambda *i: elementwise_bop(lhs[tuple(i)], rhs)
        return tvm.compute(lhs.shape, f, "tensor_" + name)
    else:
        assert False


def _make_bop(name, elementwise_bop, broadcast_bop):
    """Wrapper function to make a specific overloaded operator."""
    return lambda lhs, rhs: _tensor_bop_impl(lhs, rhs, name, elementwise_bop, broadcast_bop)


tvm.generic.add = _make_bop("add", lambda x, y: x + y, _broadcast.broadcast_add)
tvm.generic.sub = _make_bop("sub", lambda x, y: x - y, _broadcast.broadcast_sub)
tvm.generic.mul = _make_bop("mul", lambda x, y: x * y, _broadcast.broadcast_mul)
tvm.generic.div = _make_bop("div", lambda x, y: x / y, _broadcast.broadcast_div)
