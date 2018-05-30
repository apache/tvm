"""TVM operator upsampling compute."""
from __future__ import absolute_import
import topi


def upsampling(data, scale, layout="NCHW", align_corners=False):
    """Perform nearest neighbor upsampling on the data.
       Bilinear upsampling is not supported.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale: int
        upsampling scaling factor

    layout: string
        either "NCHW" or "NHWC"

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """

    if layout == "NCHW":
        out_shape = (data.shape[2] * scale, data.shape[3] * scale)
    elif layout == "NHWC":
        out_shape = (data.shape[1] * scale, data.shape[2] * scale)
    else:
        raise ValueError("not support this layout {} yet".format(layout))

    return topi.cpp.nn.scale([data], out_shape, layout, align_corners, "NN")
