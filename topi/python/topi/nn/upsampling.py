"""TVM operator upsampling compute."""
from __future__ import absolute_import
import tvm
from .. import util


def upsampling(data, scale, layout="NCHW"):
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
        return upsampling_nchw(data, scale)
    elif layout == "NHWC":
        return upsampling_nhwc(data, scale)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def upsampling_nchw(data, scale):
    """Perform nearest neighor upsampling on NCHW layout input.

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
    out_height = util.simplify(height * scale)
    out_width = util.simplify(width * scale)

    return tvm.compute((batch, channel, out_height, out_width), \
                        lambda n, c, h, w: data[n, c, h/scale, w/scale])


def upsampling_nhwc(data, scale):
    """Perform nearest neighor upsampling on NHWC layout input.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, channel]

    scale: int
        upsampling scaling factor

    """

    batch, height, width, channel = data.shape
    out_height = util.simplify(height * scale)
    out_width = util.simplify(width * scale)

    return tvm.compute((batch, out_height, out_width, channel), \
                        lambda n, h, w, c: data[n, h/scale, w/scale, c])
