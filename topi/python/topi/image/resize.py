"""TVM operator input resize compute."""
from __future__ import absolute_import
import topi

def resize(data, size, layout="NCHW", align_corners=False, method="BILINEAR"):
    """Perform resize operation on the data.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    size: Tuple
        Output resolution scale to

    layout: string, optional
        either "NCHW" or "NHWC"

    align_corners: Boolean, optional
        To preserve the values at the corner pixels

    method: {"BILINEAR", "NEAREST_NEIGHBOR"}
        Method to be used for resizing.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """
    return topi.cpp.image.resize(data, size, layout, align_corners, method)
