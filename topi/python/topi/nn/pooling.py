"""TVM operator pooling compute."""
from __future__ import absolute_import
import tvm
from .pad import pad
from .util import get_pad_tuple
from .. import util

def max_pool(data, kernel, stride, padding):
    """Perform max pooling on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    kernel : list/tuple of two ints
        Kernel size, or [kernel_height, kernel_width]

    stride : list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    paddding : list/tuple of two ints
        Pad size, or [pad_height, pad_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, out_height, out_width]
    """
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    batch, channel, height, width = data.shape

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_height, kernel_width))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp",
               pad_value=tvm.min_value("float32"))
    out_height = util.simplify((height - kernel_height + pad_top + pad_down) // stride_height + 1)
    out_width = util.simplify((width - kernel_width + pad_left + pad_right) // stride_width + 1)
    dheight = tvm.reduce_axis((0, kernel_height))
    dwidth = tvm.reduce_axis((0, kernel_width))

    return tvm.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w:
        tvm.max(temp[i, c, h*stride_height+dheight, w*stride_width+dwidth], axis=[dheight, dwidth]),
        tag="max_pool")


@tvm.tag_scope(tag='global_avg_pool')
def global_avg_pool(data):
    """Perform global average pooling on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, 1, 1]
    """
    assert len(data.shape) == 4, "only support 4-dim pooling"
    batch, channel, height, width = data.shape

    dheight = tvm.reduce_axis((0, height))
    dwidth = tvm.reduce_axis((0, width))

    tsum = tvm.compute((batch, channel, 1, 1), lambda n, c, h, w: \
        tvm.sum(data[n, c, dheight, dwidth], axis=[dheight, dwidth]))
    return tvm.compute((batch, channel, 1, 1), lambda n, c, h, w: \
        tsum[n, c, h, w] / (height*width))
