"""
REORG Operator
====================
**Author**: `Siju Samuel <https://github.com/siju-samuel/>`_
Reorg operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
import topi

def _simplify(shape):
    return int(str(shape[0])), int(str(shape[1])), int(str(shape[2])), int(str(shape[3]))

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
    batch, c, h, w = _simplify(data.shape)
    out_c = int(c / (stride * stride))
    out = tvm.compute((batch, c, h, w), lambda b, k, j, i:
                data[b * stride * stride,
                    (k % out_c) * stride * stride,
                        (j*stride + (k / out_c) / stride) * stride,
                            (i*stride + (k / out_c) % stride)],
                tag="reorg")
    out_c = int(c * stride * stride)
    out_h = int(h / stride)
    out_w = int(w / stride)
    return topi.reshape(out, (batch, out_c, out_h, out_w))
