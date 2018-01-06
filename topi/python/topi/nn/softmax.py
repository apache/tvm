# pylint: disable=invalid-name
"""TVM operator for softmax and log_softmax compute."""
from __future__ import absolute_import
import tvm

@tvm.tag_scope(tag='softmax_output')
def softmax(x, axis):
    if axis == 1 or axis == -1:
        return softmax_nchw(x, axis)
    elif axis == 3 or axis == 4:
        return softmax_nhwc(x, axis)
    else:
        raise ValueError("not support this axis {} yet".format(axis))


def softmax_nchw(x, axis):
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


def softmax_nhwc(x, axis):
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
    if len(x.shape) == 5:
        assert axis == 4, "only support 5-dim axis == 4 softmax"
        n, d, h, w, c = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, d, h, w), lambda n, d, h, w: tvm.max(x[n, d, h, w, k], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, d, h, w), lambda n, d, h, w: tvm.sum(tvm.exp(x[n, d, h, w, k] - max_elem[n, d, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, d, h, w, c: tvm.exp(x[n, d, h, w, c] - max_elem[n, d, h, w]) / expsum[n, d, h, w])
    elif len(x.shape) == 4:
        assert axis == 3
        n, h, w, c = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, h, w), lambda n, h, w: tvm.max(x[n, h, w, k], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, h, w), lambda n, h, w: tvm.sum(tvm.exp(x[n, h, w, k] - max_elem[n, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, h, w, c: tvm.exp(x[n, h, w, c] - max_elem[n, h, w]) / expsum[n, h, w])
    else:
        raise ValueError("2 dim softmax with nhwc layout not supported.")


@tvm.tag_scope(tag='log_softmax_output')
def log_softmax(x):
    """Perform log softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        2-D input data

    Returns
    -------
    output : tvm.Tensor
        2-D output with same shape
    """

    assert len(x.shape) == 2, "only support 2-dim log softmax"
    m, n = x.shape
    k = tvm.reduce_axis((0, n), name='k')
    max_elem = tvm.compute((m, ), lambda i: tvm.max(x[i, k], axis=k))
    k = tvm.reduce_axis((0, n), name='k')
    expsum = tvm.compute(
        (m, ), lambda i: tvm.sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return tvm.compute(
        x.shape, lambda i, j: x[i, j] - max_elem[i] - tvm.log(expsum[i]))
