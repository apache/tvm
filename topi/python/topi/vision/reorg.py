"""
REORG Operator
====================
Reorg operator, used in darknet.
"""
from __future__ import absolute_import as _abs
import tvm
from .. import cpp

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
    return cpp.vision.reorg(data, stride)
