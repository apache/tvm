"""TVM operator upsampling compute."""
from __future__ import absolute_import
import tvm


def upsampling(data, scale):
    """Perform nearest neighbor upsampling on the data.
       Bilinear upsampling is not supported.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    scale: int
        upsampling scaling factor

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
    """
    batch, channel, height, width = data.shape
    out_height = height * scale
    out_width = width * scale

    return tvm.compute((batch, channel, out_height, out_width), \
                        lambda n, c, h, w: data[n, c, h/scale, w/scale])
