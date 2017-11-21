"""External function interface to MPS libraroes."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin


def matmul(lhs, rhs, transa=False, transb=False):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to calle external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    m = lhs.shape[0]
    n = rhs.shape[1]
    return _api.extern(
        (n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.mps.matmul", ins[0], ins[1], outs[0], transa, transb),
        name="C")
