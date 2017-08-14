"""TVM operator pooling compute."""
from __future__ import absolute_import
import tvm

@tvm.tag_scope(tag='max_pool')
def max_pool(data, kernel, stride, pad):
    """Perform max pooling on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    kernel : list/tuple of two ints
        Kernel size, or [kernel_height, kernel_width]

    stride : list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    pad : list/tuple of two ints
        Pad size, or [pad_height, pad_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, out_height, out_width]
    """
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride.shape) == 2, "only support 2-dim stride"
    assert len(pad.shape) == 2, "only support 2-dim pad"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    pad_height, pad_width = pad
    batch, channel, height, width = data.shape
    padded_height = height + 2*pad_height
    padded_width = width + 2*pad_width
    out_height = (height + 2*pad_height - kernl_height) / stride_height + 1
    out_width = (width + 2*pad_width - kernel_width) / stride_width + 1

    dheight = tvm.reduce_axis((0, kernel_height))
    dwidth = tvm.reduce_axis((0, kernel_width))

    temp = tvm.compute((batch, channel, padded_height, padded_width), lambda i, c, h, w: \
        tvm.select(
            tvm.make.Or(tvm.make.Or((h < pad_height), (h >= height + pad_height)),
                        tvm.make.Or((w < pad_width), (w >= width + pad_width))),
            tvm.min_value('float32'),
            data[i, c, h - pad_height, w - pad_width]), name='temp')

    return tvm.compute((batch, channel, out_height, out_width), lambda i, c, h, w: \
        tvm.max(temp[i, c, h*stride_height+dheight, w*stride_width+dwidth], axis=[dheight, dwidth]))


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
