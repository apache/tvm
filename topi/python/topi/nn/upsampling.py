"""TVM operator upsampling compute."""
from __future__ import absolute_import
import topi


def upsampling(data, scale, layout="NCHW", method='NEAREST_NEIGHBOR', weights=None):
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

    weights : tvm.Tensor, optional
        weights is valid only for method=BILINEAR
        A 3-D Tensor with shape [out_shape[0], out_shape[1], 4]
        helper function tvm.contrib.image.bilinear_weights available to generate this.

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

    scale_inputs = [data, weights] if method == "BILINEAR" else [data]

    return topi.cpp.nn.upsampling(scale_inputs, out_shape, layout, method)
