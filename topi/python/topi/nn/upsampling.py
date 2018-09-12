"""TVM operator upsampling compute."""
from __future__ import absolute_import
import topi
from ..util import simplify


def upsampling(data, scale, layout="NCHW", method='NEAREST_NEIGHBOR'):
    """Perform upsampling on the data.
       Nearest neighbor and bilinear upsampling are supported.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale : int
        Scaling factor

    layout : string, optional
        either "NCHW" or "NHWC"

    method : {"BILINEAR", "NEAREST_NEIGHBOR"}
        Method to be used for upsampling.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """

    if layout == "NCHW":
        out_shape = (simplify(data.shape[2] * scale), simplify(data.shape[3] * scale))
    elif layout == "NHWC":
        out_shape = (simplify(data.shape[1] * scale), simplify(data.shape[2] * scale))
    else:
        raise ValueError("not support this layout {} yet".format(layout))

    return topi.cpp.nn.upsampling(data, out_shape, layout, method)
