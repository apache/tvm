"""TVM operator bilinear scaling compute."""
from __future__ import absolute_import
import topi


def bilinear_scale(data, weights, out_size, layout="NCHW", align_corners=False):
    """Perform bilinear scaling on the data.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    weights: tvm.Tensor
        1-D with weights [x, y, x_diff, y_diff]
        helper function tvm.contrib.image.bilinear_weights available to generate this.

    layout: string
        either "NCHW" or "NHWC"

    out_size: Tuple
        Tuple of (out_height, out_width)

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, out_height, out_width]
        or [batch, out_height, out_width, channel]
    """
    return topi.cpp.nn.scale([data, weights], out_size, layout, align_corners, "BILINEAR")
