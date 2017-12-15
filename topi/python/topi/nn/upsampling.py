"""TVM operator upsampling compute."""
from __future__ import absolute_import
import tvm
from .pad import pad
from .util import get_pad_tuple
from .. import util
from .. import tag


def upsampling(data, scale):
    """Only supports nearest neighbor for now.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    scale: int
        upsampling scaling factor

    """

    batch, channel, height, width = data.shape
    out_height = height * scale
    out_width = width * scale

    return tvm.compute((batch, channel, out_height, out_width), \
                        lambda n, c, h, w: data[n, c, h/scale, w/scale])
