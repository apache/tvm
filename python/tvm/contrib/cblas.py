"""External function interface to BLAS libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def matmul(lhs, rhs, transa=False, transb=False):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to call external libraries.

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
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    return _api.extern(
        (n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cblas.matmul",
            ins[0], ins[1], outs[0], transa, transb), name="C")
