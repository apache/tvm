# pylint: disable=invalid-name
"""TVM operator softmax compute."""
from __future__ import absolute_import
import tvm

@tvm.tag_scope(tag='softmax_output')
def softmax(x):
    """Perform softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        2-D input data

    Returns
    -------
    output : tvm.Tensor
        2-D output with same shape
    """
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = tvm.reduce_axis((0, n), name='k')
    max_elem = tvm.compute((m, ), lambda i: tvm.max(x[i, k], axis=k))
    k = tvm.reduce_axis((0, n), name='k')
    expsum = tvm.compute(
        (m, ), lambda i: tvm.sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return tvm.compute(
        x.shape, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i])
