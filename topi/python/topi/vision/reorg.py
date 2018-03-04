"""
REORG Operator
====================
Reorg operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from .. import util
from .. import transform

@tvm.target.generic_func
def reorg(data, stride):
    """Reorg forward operators.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    stride : int
        Stride value for reorganization

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, c_in, h_in, w_in = util.get_const_tuple(data.shape)
    out_c = int(c_in / (stride * stride))
    out = tvm.compute((batch, c_in, h_in, w_in), lambda b, k, j, i:
                      data[b * stride * stride,
                           (k % out_c) * stride * stride,
                           (j*stride + (k / out_c) / stride) * stride,
                           (i*stride + (k / out_c) % stride)],
                      tag="reorg")
    out_c = int(c_in * stride * stride)
    out_h = int(h_in / stride)
    out_w = int(w_in / stride)
    return transform.reshape(out, (batch, out_c, out_h, out_w))
