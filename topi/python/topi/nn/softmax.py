# pylint: disable=invalid-name
"""TVM operator for softmax and log_softmax compute."""
from __future__ import absolute_import
import tvm

@tvm.tag_scope(tag='softmax_output')
def softmax(x, axis=-1):
    """Perform softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        input can be 2d, 4d, or 5d

    axis : int
        channel axis

    Returns
    -------
    output : tvm.Tensor
        output shape is the same as input
    """
    if axis == 1:
        return softmax_nchw(x, axis)
    elif axis == 3 or axis == 4:
        return softmax_nhwc(x, axis)
    assert len(x.shape) == 2 and axis == -1, "only support 2-dim axis == -1 softmax"
    m, n = x.shape
    k = tvm.reduce_axis((0, n), name='k')
    max_elem = tvm.compute((m, ), lambda i: tvm.max(x[i, k], axis=k))
    k = tvm.reduce_axis((0, n), name='k')
    expsum = tvm.compute(
        (m, ), lambda i: tvm.sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return tvm.compute(
        x.shape, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i])


def softmax_nchw(x, axis):
    """Perform softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        4d data in NCHW layout or
        5d data in NCDHW laout

    axis : int
        channel axis

    Returns
    -------
    output : tvm.Tensor
        output shape is the same as input
    """
    if len(x.shape) == 5:
        assert axis == 1, "only support 5-dim axis == 1 softmax for NCDHW layout"
        n, c, d, h, w = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, d, h, w), lambda n, d, h, w: tvm.max(x[n, k, d, h, w], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, d, h, w), lambda n, d, h, w: tvm.sum(tvm.exp(x[n, k, d, h, w] - max_elem[n, d, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, c, d, h, w: tvm.exp(x[n, c, d, h, w] - max_elem[n, d, h, w]) / expsum[n, d, h, w])
    elif len(x.shape) == 4:
        assert axis == 1, "only support 4-dim axis == 1 softmax for NCHW layout"
        n, c, h, w = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, h, w), lambda n, h, w: tvm.max(x[n, k, h, w], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, h, w), lambda n, h, w: tvm.sum(tvm.exp(x[n, k, h, w] - max_elem[n, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, c, h, w: tvm.exp(x[n, c, h, w] - max_elem[n, h, w]) / expsum[n, h, w])


def softmax_nhwc(x, axis):
    """Perform softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        4d data in NHWC layout or
        5d data in NDHWC laout

    axis : int
        channel axis

    Returns
    -------
    output : tvm.Tensor
        output shape is the same as input
    """
    if len(x.shape) == 5:
        assert axis == 4, "only support 5-dim axis == 4 softmax for NDHWC layout"
        n, d, h, w, c = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, d, h, w), lambda n, d, h, w: tvm.max(x[n, d, h, w, k], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, d, h, w), lambda n, d, h, w: tvm.sum(tvm.exp(x[n, d, h, w, k] - max_elem[n, d, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, d, h, w, c: tvm.exp(x[n, d, h, w, c] - max_elem[n, d, h, w]) / expsum[n, d, h, w])
    elif len(x.shape) == 4:
        assert axis == 3, "only support 4-dim axis == 3 softmax for NHWC layout"
        n, h, w, c = x.shape
        k = tvm.reduce_axis((0, c), name='k')
        max_elem = tvm.compute((n, h, w), lambda n, h, w: tvm.max(x[n, h, w, k], axis=k))
        k = tvm.reduce_axis((0, c), name='k')
        expsum = tvm.compute(
            (n, h, w), lambda n, h, w: tvm.sum(tvm.exp(x[n, h, w, k] - max_elem[n, h, w]), axis=k))
        return tvm.compute(
            x.shape, lambda n, h, w, c: tvm.exp(x[n, h, w, c] - max_elem[n, h, w]) / expsum[n, h, w])


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
