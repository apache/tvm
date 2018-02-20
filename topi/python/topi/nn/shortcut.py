"""Shortcut operators (short-cut connections)."""
from __future__ import absolute_import as _abs
import tvm
import topi

def _simplify(shape):
    return int(str(shape[0])), int(str(shape[1])), int(str(shape[2])), int(str(shape[3]))

@tvm.target.generic_func
def shortcut(inp1, inp2):
    """Shortcut forward operators.

    Parameters
    ----------
    First Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Second Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    _, c1, h1, w1 = _simplify(inp1.shape)
    batch, c2, h2, w2 = _simplify(inp2.shape)

    stride = int (max(w2 / w1, 1))
    sample = int(max(w1 / w2, 1))
    minc   = min(c2, c1)
    minh   = min(h2, h1)
    minw   = min(w2, w1)

    out = tvm.compute((batch, minc, minh, minw), lambda b, c, h, w:
            inp1[b, c, h * sample, w * sample] +
                inp2[b, c, h * stride, w * stride],
                    tag="shortcut")

    split_indices = int(c1 / minc)
    if split_indices > 1:
        split_res = topi.split(inp1, split_indices, 1)
        split_res[0] = out
        out = topi.concatenate(split_res, 1)

    return out

