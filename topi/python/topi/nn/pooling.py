"""TVM operator pooling compute."""
from __future__ import absolute_import
import tvm
from .pad import pad
from .util import get_pad_tuple
from .. import util
from .. import tag


def global_pool(data, pool_type):
    """Perform global pooling on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    pool_type : str
        Pool type, 'max' or 'avg'

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, 1, 1]
    """
    assert len(data.shape) == 4, "only support 4-dim pooling"
    batch, channel, height, width = data.shape

    dheight = tvm.reduce_axis((0, height))
    dwidth = tvm.reduce_axis((0, width))

    if pool_type == 'max':
        return tvm.compute((batch, channel, 1, 1), lambda n, c, h, w: \
                            tvm.max(data[n, c, dheight, dwidth], axis=[dheight, dwidth]), \
                            tag="global_pool_max")
    elif pool_type == 'avg':
        tsum = tvm.compute((batch, channel, 1, 1), lambda n, c, h, w: \
                            tvm.sum(data[n, c, dheight, dwidth], axis=[dheight, dwidth]), \
                            tag="global_pool_sum")
        return tvm.compute((batch, channel, 1, 1), lambda n, c, h, w: \
                            tsum[n, c, h, w] / (height*width).astype(tsum.dtype), \
                            tag=tag.ELEMWISE)
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")


def pool(data, kernel, stride, padding, pool_type):
    """Perform pooling on the data

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    kernel : list/tuple of two ints
        Kernel size, [kernel_height, kernel_width]

    stride : list/tuple of two ints
        Stride size, [stride_height, stride_width]

    paddding : list/tuple of two ints
        Pad size, [pad_height, pad_width]

    pool_type : str
        Pool type, 'max' or 'avg'

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
    out_height = util.simplify((height - kernel_height + pad_top + pad_down) // stride_height + 1)
    out_width = util.simplify((width - kernel_width + pad_left + pad_right) // stride_width + 1)
    dheight = tvm.reduce_axis((0, kernel_height))
    dwidth = tvm.reduce_axis((0, kernel_width))

    if pool_type == 'max':
        temp = pad(data, pad_before, pad_after, name="pad_temp", \
            pad_value=tvm.min_value(data.dtype))
        return tvm.compute((batch, channel, out_height, out_width), \
                            lambda n, c, h, w: \
                            tvm.max(temp[n, c, h*stride_height+dheight, w*stride_width+dwidth], \
                                axis=[dheight, dwidth]), \
                            tag="pool_max")
    elif pool_type == 'avg':
        temp = pad(data, pad_before, pad_after, name="pad_temp", \
            pad_value=tvm.const(0.).astype(data.dtype))
        tsum = tvm.compute((batch, channel, out_height, out_width), \
                            lambda n, c, h, w: \
                            tvm.sum(temp[n, c, h*stride_height+dheight, w*stride_width+dwidth], \
                                axis=[dheight, dwidth]), \
                            tag="pool_avg")
        return tvm.compute((batch, channel, out_height, out_width), \
                            lambda n, c, h, w: \
                            tsum[n, c, h, w] / (kernel_height*kernel_width), \
                            tag=tag.ELEMWISE)
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")
