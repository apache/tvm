"""Implementation of generic operators in the presence of Tensor"""
# pylint: disable=invalid-name, too-many-arguments
from __future__ import absolute_import as _abs
import tvm
from . import broadcast as _broadcast
from . import tag


def _make_bop(elementwise_bop, broadcast_bop, orig_bop):
    """Make a specific overloaded binary operator of Tensor when applicable;
    apply the original operator if it is not supposed to be overloaded.

    Consider the following scenario:
    OP :   + | - | * | /
    R0 :   int | float | Expr | TensorSlice | Tensor (rank zero)
    R1 :   Tensor (positive rank)

    In terms of (LHS OP RHS), we apply the following overloading rules:
    (1) We use broadcast_OP(LHS, RHS), when both LHS and RHS are R1.
    (2) We perform element-wise operation of Tensor and scalar,
        when one of LHS and RHS is R1 and another is R0.
    (3) We do not overload OP (i.e. stick to orig_bop) otherwise.

    Parameters
    ----------
    elementwise_bop : operator function
        Operator for element-wise tensor-scalar operation, for rule (2).

    broadcast_bop : operator function
        Operator for broadcast tensor-tensor operation, for rule (1).

    orig_bop: operator function
        Operator before overloading, for rule (3).

    Returns
    -------
    ret : operator function
        The overloaded operator function if applicable or orig_bop otherwise.
    """

    name = orig_bop.__name__

    def _tensor_bop_impl(lhs, rhs):
        """Overloaded {op} operator.

        If both operands are non-zero-rank Tensors, it performs
        tensor-tensor {op} operation, and broadcasts inputs when necessary.

        If one operand is non-zero-rank Tensor, while the other operand is
        scalar like type (e.g., numeric types, Expr, or TensorSlice),
        it performs tensor-scalar {op} operation on an element-wise basis.

        Otherwise, it performs default generic.{op} operation, as defined
        in tvm.generic module.

        Parameters
        ----------
        lhs : object
            Left operand.
        rhs : object
            Right operand.

        Returns
        -------
        ret : tvm.Tensor (if at least one operand is non-zero-rank Tensor)
              tvm.Expr (otherwise)
            The result of {op} operation.
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
            return orig_bop(lhs, rhs)
        elif rl > 0 and rr > 0:
            return broadcast_bop(lhs, rhs)
        elif rl == 0:
            f = lambda *i: elementwise_bop(lhs, rhs(*i))
            with tvm.tag_scope(tag=tag.ELEMWISE):
                return tvm.compute(rhs.shape, f, "tensor_" + name)
        elif rr == 0:
            f = lambda *i: elementwise_bop(lhs(*i), rhs)
            with tvm.tag_scope(tag=tag.ELEMWISE):
                return tvm.compute(lhs.shape, f, "tensor_" + name)
        else:
            raise AssertionError("Cannot reach this line.")

    _tensor_bop_impl.__doc__ = _tensor_bop_impl.__doc__.format(op=name)
    return _tensor_bop_impl


def _bind_generic_ops():
    """Bind generic operators for Tensor."""
    # Check __op_priority__ to make sure the binding happens only once.
    __op_priority__ = 1
    if __op_priority__ > tvm.generic.__op_priority__:
        tvm.generic.__op_priority__ = __op_priority__
        tvm.generic.add = _make_bop(lambda x, y: x + y,
                                    _broadcast.broadcast_add,
                                    tvm.generic.add)
        tvm.generic.subtract = _make_bop(lambda x, y: x - y,
                                         _broadcast.broadcast_sub,
                                         tvm.generic.subtract)
        tvm.generic.multiply = _make_bop(lambda x, y: x * y,
                                         _broadcast.broadcast_mul,
                                         tvm.generic.multiply)
        tvm.generic.divide = _make_bop(lambda x, y: x / y,
                                       _broadcast.broadcast_div,
                                       tvm.generic.divide)


_bind_generic_ops()
