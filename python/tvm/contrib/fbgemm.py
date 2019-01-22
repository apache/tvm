"""External function interface to FBGEMM libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api


def matmul_fp16(A, B, nthreads=1):
    """Create an extern op that compute matrix multiply with fbgemm fp16.

    Parameters
    ----------
    A : Tensor
        2D array M*K
    B : Tensor
        2D array K*N

    Returns
    -------
    C : Tensor
        2D array out M*N
    """
    n = B.shape[1]
    m = A.shape[0]
    return _api.extern(
        (m, n), [A, B],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.fbgemm.matmul_fp16",
            ins[0], ins[1], outs[0], nthreads), name="C")


_init_api("tvm.contrib.fbgemm")
