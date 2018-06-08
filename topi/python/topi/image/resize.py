"""TVM operator input resize compute."""
from __future__ import absolute_import
import topi

def resize(data, out_size, layout="NCHW", align_corners=False, mode="BILINEAR", weights=None):
    """Perform resize operation on the data.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    out_shape: Tuple
        Output resolution scale to

    layout: string
        either "NCHW" or "NHWC"

    align_corners: Boolean
        To preserve the values at the corner pixels

    mode: string
        either "NN" or "BILINEAR"

    weights:
        weights is valid only for mode=BILINEAR
        A 4-D Tensor with shape [out_shape[0], out_shape[1], 4]
        helper function tvm.contrib.image.bilinear_weights available to generate this.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """
    inputs = [data, weights] if mode == "BILINEAR" else [data]

    return topi.cpp.image.resize(inputs, out_size, layout, align_corners, mode)
